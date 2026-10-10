// fastfft::cx -- the fastfft kernel written against C++26 std::simd::vec<std::complex<T>> (experimental).
//
// Same algorithm as fft.hpp (radix-8 first pass vectorised over the butterfly index with an in-register interleave,
// radix-16 Stockham passes vectorised over the stride, final radix 2/4/8), but every vector is a vector of complex
// numbers, so complex multiplication, the +-i rotations and the twiddle tables are expressed as complex arithmetic
// instead of re/im lane tricks. Needs GCC >= 17 (libstdc++'s <simd> with simd_complex) and -std=c++26; compile with
// -fcx-limited-range (otherwise operator* carries the Annex G NaN/Inf recovery path) and -ffp-contract=fast.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef FASTFFT_FFT_CX_HPP
#define FASTFFT_FFT_CX_HPP

#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <memory>
#include <numbers>
#include <simd>
#include <stdexcept>
#include <vector>

namespace fastfft::cx {

namespace detail {

template<std::floating_point T>
struct Ops {
    using Cx = std::complex<T>;
#ifdef FASTFFT_SIMD_BYTES
    static constexpr int C = FASTFFT_SIMD_BYTES / int(sizeof(Cx)) < 2 ? 2 : FASTFFT_SIMD_BYTES / int(sizeof(Cx));
#else
    static constexpr int C = std::simd::vec<T>::size() / 2 < 2 ? 2 : std::simd::vec<T>::size() / 2; // complex lanes
#endif
    using V = std::simd::vec<Cx, C>;
    using R = std::simd::vec<T, C>;

    [[gnu::always_inline]] static V load(const Cx* p) { return std::simd::unchecked_load<V>(p, C); }
    [[gnu::always_inline]] static void store(const V& v, Cx* p) { std::simd::unchecked_store(v, p, C); }
    // broadcast; V(*p) currently compiles to scalar stores, building it from real/imag broadcasts does not
#ifdef FASTFFT_CX_EXPERIMENT_BITCAST
    [[gnu::always_inline]] static V bcast(const Cx* p) { double d; std::memcpy(&d, p, 8); if constexpr (sizeof(T) == 4) return std::bit_cast<V>(std::simd::vec<double, C>(d)); else return V(R(p->real()), R(p->imag())); }
#else
    [[gnu::always_inline]] static V bcast(const Cx* p) { return V(R(p->real()), R(p->imag())); }
#endif

    [[gnu::always_inline]] static V mulMinusI(const V& v) { return v * V(Cx(0, -1)); }
    [[gnu::always_inline]] static V mulPlusI(const V& v) { return v * V(Cx(0, 1)); }
    [[gnu::always_inline]] static V mulW8(const V& v) { constexpr T r = T(0.70710678118654752); return v * V(Cx(r, -r)); }
    [[gnu::always_inline]] static V mulW8_3(const V& v) { constexpr T r = T(0.70710678118654752); return v * V(Cx(-r, -r)); }

    [[gnu::always_inline]] static void bfly4(const V& a, const V& b, const V& c, const V& d, V& y0, V& y1, V& y2, V& y3) {
        const V apc = a + c, amc = a - c, bpd = b + d, jbmd = mulMinusI(b - d);
        y0 = apc + bpd;
        y2 = apc - bpd;
        y1 = amc + jbmd;
        y3 = amc - jbmd;
    }

    [[gnu::always_inline]] static void bfly8(const V* a, V* y) {
        const V b0 = a[0] + a[4], b1 = a[1] + a[5], b2 = a[2] + a[6], b3 = a[3] + a[7];
        const V c0 = a[0] - a[4], c1 = mulW8(a[1] - a[5]), c2 = mulMinusI(a[2] - a[6]), c3 = mulW8_3(a[3] - a[7]);
        bfly4(b0, b1, b2, b3, y[0], y[2], y[4], y[6]);
        bfly4(c0, c1, c2, c3, y[1], y[3], y[5], y[7]);
    }

    [[gnu::always_inline]] static void bfly16(const V* a, V* y) {
        constexpr T c1 = T(0.92387953251128674), s1 = T(0.38268343236508977);
        V           B[4][4];
        for (int j1 = 0; j1 < 4; ++j1) {
            bfly4(a[j1], a[j1 + 4], a[j1 + 8], a[j1 + 12], B[j1][0], B[j1][1], B[j1][2], B[j1][3]);
        }
        B[1][1] = B[1][1] * V(Cx(c1, -s1)); // W16^1
        B[1][2] = mulW8(B[1][2]);
        B[1][3] = B[1][3] * V(Cx(s1, -c1)); // W16^3
        B[2][1] = mulW8(B[2][1]);
        B[2][2] = mulMinusI(B[2][2]);
        B[2][3] = mulW8_3(B[2][3]);
        B[3][1] = B[3][1] * V(Cx(s1, -c1));
        B[3][2] = mulW8_3(B[3][2]);
        B[3][3] = B[3][3] * V(Cx(-c1, s1)); // W16^9
        for (int k = 0; k < 4; ++k) {
            bfly4(B[0][k], B[1][k], B[2][k], B[3][k], y[k], y[k + 4], y[k + 8], y[k + 12]);
        }
    }

    // complex-lane zip: lo = (a0 b0 a1 b1 ...), hi = the second half. permute() does not accept complex vectors yet,
    // so this goes through real()/imag(), a real-valued permute of the concatenation, and the (re, im) constructor.
    template<bool Hi>
    [[gnu::always_inline]] static V zip(const V& a, const V& b) {
        constexpr auto idx = [](int j) { return (j & 1) * C + j / 2 + (Hi ? C / 2 : 0); };
#ifdef FASTFFT_CX_EXPERIMENT_BITCAST // attribution experiment only: relies on the interleaved layout
        using F = std::simd::vec<T, 2 * C>;
        constexpr auto fidx = [](int j) { const int jc = j / 2; return (jc & 1) * 2 * C + 2 * (jc / 2 + (Hi ? C / 2 : 0)) + (j & 1); };
        return std::bit_cast<V>(std::simd::permute<2 * C>(std::simd::cat(std::bit_cast<F>(a), std::bit_cast<F>(b)), fidx));
#endif
        return V(std::simd::permute<C>(std::simd::cat(a.real(), b.real()), idx), std::simd::permute<C>(std::simd::cat(a.imag(), b.imag()), idx));
    }

    // interleave 8 vectors s[k] (lane p) into out[8p + k] order, as 8 consecutive vectors
    [[gnu::always_inline]] static void interleave8(const V* s, V* out) {
        V             r1[8], r2[8];
        constexpr int A[4] = {0, 2, 1, 3};
        for (int i = 0; i < 4; ++i) {
            r1[2 * i]     = zip<false>(s[A[i]], s[A[i] + 4]);
            r1[2 * i + 1] = zip<true>(s[A[i]], s[A[i] + 4]);
        }
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < 2; ++i) {
                r2[4 * h + 2 * i]     = zip<false>(r1[4 * h + i], r1[4 * h + 2 + i]);
                r2[4 * h + 2 * i + 1] = zip<true>(r1[4 * h + i], r1[4 * h + 2 + i]);
            }
        }
        for (int i = 0; i < 4; ++i) {
            out[2 * i]     = zip<false>(r2[i], r2[4 + i]);
            out[2 * i + 1] = zip<true>(r2[i], r2[4 + i]);
        }
    }
};

struct AlignedFree {
    void operator()(void* p) const noexcept { std::free(p); }
};

} // namespace detail

template<std::floating_point T>
class FFT {
    using Ops = detail::Ops<T>;
    using V   = typename Ops::V;
    using Cx  = std::complex<T>;
    static constexpr std::size_t C = Ops::C;

    struct Stage {
        std::size_t n, s, m, radix, tw;
    };

public:
    FFT() = default;
    explicit FFT(std::size_t N) { init(N); }
    std::size_t size() const noexcept { return _N; }

    void init(std::size_t N) {
        if (N < 2 || !std::has_single_bit(N)) {
            throw std::invalid_argument("fastfft::cx::FFT: size must be a power of two >= 2");
        }
        _N = N;
        _stages.clear();
        std::vector<Cx> tw;
        auto            w = [](std::size_t k, std::size_t nn) { const double a = -2.0 * std::numbers::pi * double(k) / double(nn); return Cx(T(std::cos(a)), T(std::sin(a))); };
        _vector           = N >= 64 && (N / 8) % C == 0;
        if (_vector) {
            std::size_t n = N, s = 1;
            while (n > 1) {
                const std::size_t r = (s > 1 && n >= 16) ? 16 : ((n >= 8) ? 8 : n);
                const std::size_t m = n / r;
                _stages.push_back({n, s, m, r, tw.size()});
                if (s == 1) { // per vector of C p's: (r-1) vectors of twiddles
                    for (std::size_t v = 0; v < m / C; ++v) {
                        for (std::size_t k = 1; k < r; ++k) {
                            for (std::size_t j = 0; j < C; ++j) {
                                tw.push_back(w(k * (C * v + j), n));
                            }
                        }
                    }
                } else if (r >= 8) { // per p: (r-1) scalar twiddles, broadcast
                    for (std::size_t p = 0; p < m; ++p) {
                        for (std::size_t k = 1; k < r; ++k) {
                            tw.push_back(w(k * p, n));
                        }
                    }
                }
                n /= r;
                s *= r;
            }
        }
        _tw = std::move(tw);
        _work.reset(static_cast<Cx*>(std::aligned_alloc(64, ((N * sizeof(Cx) + 63) / 64) * 64)));
        _inCopy.reset(static_cast<Cx*>(std::aligned_alloc(64, ((N * sizeof(Cx) + 63) / 64) * 64)));
    }

    /// out[k] = sum_n in[n] exp(-2 pi i n k / N); in == out is allowed, no alignment requirement
    void forward(const Cx* in, Cx* out) const {
        if (!_vector) {
            scalarForward(in, out);
            return;
        }
        if (in == out) {
            std::memcpy(_inCopy.get(), in, _N * sizeof(Cx));
            in = _inCopy.get();
        }
        const std::size_t nS  = _stages.size();
        const Cx*         src = in;
        for (std::size_t i = 0; i < nS; ++i) {
            Cx* dst = ((nS - 1 - i) % 2 == 0) ? out : _work.get();
            runStage(_stages[i], src, dst);
            src = dst;
        }
    }

private:
    std::size_t                                  _N{0};
    bool                                         _vector{false};
    std::vector<Stage>                           _stages;
    std::vector<Cx>                              _tw;
    std::unique_ptr<Cx, detail::AlignedFree> _work, _inCopy;

    void runStage(const Stage& st, const Cx* x, Cx* y) const {
        const std::size_t m = st.m, s = st.s;
        if (st.radix == 2) {
            for (std::size_t q = 0; q < s; q += C) {
                const V a = Ops::load(x + q), b = Ops::load(x + q + s);
                Ops::store(a + b, y + q);
                Ops::store(a - b, y + q + s);
            }
            return;
        }
        if (st.radix == 4) {
            for (std::size_t q = 0; q < s; q += C) {
                V y0, y1, y2, y3;
                Ops::bfly4(Ops::load(x + q), Ops::load(x + q + s), Ops::load(x + q + 2 * s), Ops::load(x + q + 3 * s), y0, y1, y2, y3);
                Ops::store(y0, y + q);
                Ops::store(y1, y + q + s);
                Ops::store(y2, y + q + 2 * s);
                Ops::store(y3, y + q + 3 * s);
            }
            return;
        }
        if (s == 1) { // radix 8, vectorised over p, interleaved on the store
            const Cx* t = _tw.data() + st.tw;
            for (std::size_t p0 = 0; p0 < m; p0 += C, t += 7 * C) {
                V a[8], o[8], r[8];
                for (std::size_t k = 0; k < 8; ++k) {
                    a[k] = Ops::load(x + p0 + k * m);
                }
                Ops::bfly8(a, o);
                for (std::size_t k = 1; k < 8; ++k) {
                    o[k] = o[k] * Ops::load(t + C * (k - 1));
                }
                Ops::interleave8(o, r);
                for (std::size_t i = 0; i < 8; ++i) {
                    Ops::store(r[i], y + 8 * p0 + C * i);
                }
            }
            return;
        }
        if (st.radix == 16) {
            middleStage<16>(st, x, y);
        } else {
            middleStage<8>(st, x, y);
        }
    }

    template<std::size_t r> // radix 8 or 16, vectorised over the stride, broadcast twiddles
    void middleStage(const Stage& st, const Cx* x, Cx* y) const {
        const std::size_t m = st.m, s = st.s;
        for (std::size_t p = 0; p < m; ++p) {
            const Cx* t  = _tw.data() + st.tw + (r - 1) * p;
            const Cx* xp = x + s * p;
            Cx*       yp = y + s * r * p;
            for (std::size_t q = 0; q < s; q += C) {
                V a[r], o[r];
                for (std::size_t k = 0; k < r; ++k) {
                    a[k] = Ops::load(xp + q + k * s * m);
                }
                if constexpr (r == 16) {
                    Ops::bfly16(a, o);
                } else {
                    Ops::bfly8(a, o);
                }
                Ops::store(o[0], yp + q);
                for (std::size_t k = 1; k < r; ++k) {
                    Ops::store(p == 0 ? o[k] : o[k] * Ops::bcast(t + k - 1), yp + q + k * s);
                }
            }
        }
    }

    void scalarForward(const Cx* in, Cx* out) const {
        const std::size_t N = _N;
        std::vector<Cx>   a(in, in + N), b(N);
        std::size_t       n = N, s = 1;
        Cx *              x = a.data(), *y = b.data();
        while (n > 1) {
            const std::size_t m = n / 2;
            for (std::size_t p = 0; p < m; ++p) {
                const double ang = -2.0 * std::numbers::pi * double(p) / double(n);
                const Cx     wp(T(std::cos(ang)), T(std::sin(ang)));
                for (std::size_t q = 0; q < s; ++q) {
                    const Cx u = x[q + s * p], v = x[q + s * (p + m)];
                    y[q + s * (2 * p)]     = u + v;
                    y[q + s * (2 * p + 1)] = (u - v) * wp;
                }
            }
            std::swap(x, y);
            n /= 2;
            s *= 2;
        }
        std::copy(x, x + N, out);
    }
};

} // namespace fastfft::cx

#endif // FASTFFT_FFT_CX_HPP
