// fastfft -- a portable std::simd FFT (complex, power-of-two sizes, float and double).
//
// Radix-8 first stage (vectorised over the butterfly index, results interleaved in registers on the store) followed
// by radix-16 Stockham autosort stages (vectorised over the stride, broadcast twiddles) and a final radix-2/-4/-8
// stage. Interleaved complex data, out-of-place or in-place, natural order in and out, unnormalised forward
// transform X[k] = sum_n x[n] exp(-2 pi i n k / N).
//
// No ISA intrinsics; the vector width follows the target. Backend, first available wins: C++26 std::simd (-std=c++26,
// libstdc++ from GCC >= 17; define FASTFFT_NO_STD_SIMD to skip it); libstdc++'s <experimental/simd> (GCC >= 12, or
// clang with -stdlib=libstdc++), which lacks a two-input shuffle, so the zip and the re/im swap use the compiler's
// generic vector extension (__builtin_shufflevector); else vir-simd (as used by gnuradio4). NOTE: on clang with
// libc++, vir-simd falls back to an array-based implementation that is several times slower (FASTFFT_NATIVE_SIMD 0).
//
// Compile with -ffp-contract=fast (the GNU-mode default of GCC) so that a*b+c becomes an FMA.
// SPDX-FileCopyrightText: 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH
// SPDX-License-Identifier: LGPL-3.0-or-later WITH LGPL-3.0-linking-exception
#ifndef FASTFFT_FFT_HPP
#define FASTFFT_FFT_HPP

#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#if __cplusplus > 202302L && __has_include(<simd>) && !defined(FASTFFT_NO_STD_SIMD)
#include <simd>
#endif
#if !defined(FASTFFT_NO_STD_SIMD) && (defined(__glibcxx_simd) || defined(__cpp_lib_simd)) // C++26 std::simd: no compiler extensions needed
#define FASTFFT_STD_SIMD     1
#define FASTFFT_SIMD_BACKEND "C++26 std::simd"
#define FASTFFT_NATIVE_SIMD  1
#elif __has_include(<experimental/simd>) && !defined(_LIBCPP_VERSION)
#include <experimental/simd>
#define FASTFFT_SIMD_BACKEND "std::experimental::simd (libstdc++)"
#define FASTFFT_NATIVE_SIMD  1 // 0: the simd types are emulated with arrays and fastfft is slow
namespace fastfft::detail {
namespace stdx = std::experimental;
}
#elif __has_include(<vir/simd.h>)
#include <vir/simd.h>
#if defined(VIR_HAVE_STD_SIMD)
#define FASTFFT_SIMD_BACKEND "vir::stdx -> std::experimental::simd"
#define FASTFFT_NATIVE_SIMD  1
#else
#define FASTFFT_SIMD_BACKEND "vir::stdx fallback (slow)"
#define FASTFFT_NATIVE_SIMD  0
#endif
namespace fastfft::detail {
namespace stdx = vir::stdx;
}
#else
#error "fastfft needs <experimental/simd> (libstdc++) or vir-simd"
#endif

namespace fastfft {

namespace detail {

template<std::floating_point T>
struct Ops {
#ifdef FASTFFT_STD_SIMD
#ifdef FASTFFT_SIMD_BYTES // benchmarking aid: force the vector size in bytes (16/32/64) instead of the native one
    static constexpr std::size_t kLanes = FASTFFT_SIMD_BYTES / sizeof(T);
#else
    static constexpr std::size_t kLanes = std::simd::vec<T>::size();
#endif
    using V = std::simd::vec<T, (kLanes < 4 ? 4 : kLanes)>; // at least two complex values per vector
#else
#ifdef FASTFFT_SIMD_BYTES // benchmarking aid: force the vector size in bytes (16/32/64) instead of the native one
    static constexpr std::size_t kLanes = FASTFFT_SIMD_BYTES / sizeof(T);
#else
    static constexpr std::size_t kLanes = stdx::native_simd<T>::size();
#endif
    // at least two complex values per vector: with 16-byte registers (SSE, NEON) double uses a pair of them
    using V = std::conditional_t<kLanes == stdx::native_simd<T>::size() && kLanes >= 4, stdx::native_simd<T>, stdx::simd<T, stdx::simd_abi::deduce_t<T, (kLanes < 4 ? 4 : kLanes)>>>;
#endif
    static constexpr std::size_t W = V::size(); // scalars per vector
    static constexpr std::size_t C = W / 2;     // complex values per vector
    static_assert(C >= 2 && C <= 8, "fastfft needs 2..8 complex values per SIMD vector");

    // a * b + c; spelled out (with -ffp-contract=fast this is one FMA) because stdx::fma is not inlined by libstdc++
    [[gnu::always_inline]] static V fma(const V& a, const V& b, const V& c) { return a * b + c; }

#ifdef FASTFFT_STD_SIMD
    [[gnu::always_inline]] static V load(const T* p) { return std::simd::unchecked_load<V>(p, W); }
    [[gnu::always_inline]] static void store(const V& v, T* p) { std::simd::unchecked_store(v, p, W); }
#else
    [[gnu::always_inline]] static V load(const T* p) { return V(p, stdx::element_aligned); }
    [[gnu::always_inline]] static void store(const V& v, T* p) { v.copy_to(p, stdx::element_aligned); }
#endif

    [[gnu::always_inline]] static V pm() { return V([](auto i) { return (i & 1) ? T(1) : -T(1); }); } // (-1, +1, ...)
    [[gnu::always_inline]] static V mp() { return V([](auto i) { return (i & 1) ? -T(1) : T(1); }); } // (+1, -1, ...)

#ifndef FASTFFT_STD_SIMD
    typedef T VB __attribute__((vector_size(sizeof(T) * W))); // the compiler's generic vector type of the same width
    // V <-> VB; a no-op for native simd types, a copy for the two-register double case of 16-byte targets
    [[gnu::always_inline]] static VB toVB(const V& v) {
        if constexpr (std::is_trivially_copyable_v<V> && sizeof(V) == sizeof(VB)) {
            return std::bit_cast<VB>(v);
        } else {
            VB r;
            v.copy_to(reinterpret_cast<T*>(&r), stdx::element_aligned);
            return r;
        }
    }
    [[gnu::always_inline]] static V fromVB(const VB& x) {
        if constexpr (std::is_trivially_copyable_v<V> && sizeof(V) == sizeof(VB)) {
            return std::bit_cast<V>(x);
        } else {
            return V(reinterpret_cast<const T*>(&x), stdx::element_aligned);
        }
    }

    template<std::size_t... J>
    [[gnu::always_inline]] static V swapImpl(const V& v, std::index_sequence<J...>) {
        const VB x = toVB(v);
        return fromVB(__builtin_shufflevector(x, x, int(J ^ 1U)...));
    }
#endif
#ifdef FASTFFT_STD_SIMD
    [[gnu::always_inline]] static V swapReIm(const V& v) { return std::simd::permute(v, [](int i) { return i ^ 1; }); }
#else
    [[gnu::always_inline]] static V swapReIm(const V& v) { return swapImpl(v, std::make_index_sequence<W>{}); }
#endif

    // broadcast one 64-bit pair of floats to every complex lane
    [[gnu::always_inline]] static V bcast2(const T* p) {
        if constexpr (sizeof(T) == 4) { // one 64-bit broadcast
#ifdef FASTFFT_STD_SIMD
            using VD = std::simd::vec<double, W / 2>;
#else
            using VD = stdx::simd<double, stdx::simd_abi::deduce_t<double, W / 2>>;
#endif
            double d;
            std::memcpy(&d, p, sizeof(d));
            return std::bit_cast<V>(VD(d));
        } else { // a generator compiles to a 128-bit broadcast here; a shufflevector of a 2-lane vector was 2x slower
            return V([p](auto i) { return p[i & 1]; });
        }
    }

    // z * w, with w given as (wr, wr) and (-wi, wi) per lane
    [[gnu::always_inline]] static V cmul(const V& z, const V& wr, const V& wiS) { return fma(z, wr, swapReIm(z) * wiS); }

    [[gnu::always_inline]] static void bfly4(const V& a, const V& b, const V& c, const V& d, V& y0, V& y1, V& y2, V& y3) {
        const V apc = a + c, amc = a - c, bpd = b + d;
        const V sw  = swapReIm(b - d);
        y0          = apc + bpd;
        y2          = apc - bpd;
        y1          = fma(sw, mp(), amc); // amc - j(b-d)
        y3          = fma(sw, pm(), amc); // amc + j(b-d)
    }

    [[gnu::always_inline]] static V negJ(const V& v) { return swapReIm(v) * mp(); }
    [[gnu::always_inline]] static V mulW8(const V& v) {
        constexpr T r = T(0.70710678118654752);
        return fma(swapReIm(v), mp() * r, v * r);
    }
    [[gnu::always_inline]] static V mulW8_3(const V& v) {
        constexpr T r = T(0.70710678118654752);
        return fma(swapReIm(v), mp() * r, v * -r);
    }

    [[gnu::always_inline]] static void bfly8(const V* a, V* y) {
        const V b0 = a[0] + a[4], b1 = a[1] + a[5], b2 = a[2] + a[6], b3 = a[3] + a[7];
        const V c0 = a[0] - a[4], c1 = mulW8(a[1] - a[5]), c2 = negJ(a[2] - a[6]), c3 = mulW8_3(a[3] - a[7]);
        bfly4(b0, b1, b2, b3, y[0], y[2], y[4], y[6]);
        bfly4(c0, c1, c2, c3, y[1], y[3], y[5], y[7]);
    }

    [[gnu::always_inline]] static V cconst(T re) { return V(re); }
    [[gnu::always_inline]] static V cconstS(T im) { return pm() * im; }

    [[gnu::always_inline]] static void bfly16(const V* a, V* y) {
        constexpr T c1 = T(0.92387953251128674), s1 = T(0.38268343236508977);
        V               B[4][4];
        for (int j1 = 0; j1 < 4; ++j1) {
            bfly4(a[j1], a[j1 + 4], a[j1 + 8], a[j1 + 12], B[j1][0], B[j1][1], B[j1][2], B[j1][3]);
        }
        B[1][1] = cmul(B[1][1], cconst(c1), cconstS(-s1)); // W16^1
        B[1][2] = mulW8(B[1][2]);
        B[1][3] = cmul(B[1][3], cconst(s1), cconstS(-c1)); // W16^3
        B[2][1] = mulW8(B[2][1]);
        B[2][2] = negJ(B[2][2]);
        B[2][3] = mulW8_3(B[2][3]);
        B[3][1] = cmul(B[3][1], cconst(s1), cconstS(-c1));
        B[3][2] = mulW8_3(B[3][2]);
        B[3][3] = cmul(B[3][3], cconst(-c1), cconstS(s1)); // W16^9
        for (int k = 0; k < 4; ++k) {
            bfly4(B[0][k], B[1][k], B[2][k], B[3][k], y[k], y[k + 4], y[k + 8], y[k + 12]);
        }
    }

    // complex-granular zip: lo = (a0 b0 a1 b1 ...) first half, hi = second half.
    // C++26 std::simd: permute of cat(a, b) lowers to one two-source shuffle (vpermt2ps on AVX-512).
    // std::experimental::simd has no two-input shuffle and permuting a concatenation gets lowered to element moves, so
    // there this helper drops to the compiler's generic vector extension (no ISA intrinsics, still width-generic).
#ifdef FASTFFT_STD_SIMD // C++26: one two-source permute
    template<bool Hi>
    [[gnu::always_inline]] static V zip(const V& a, const V& b) {
        return std::simd::permute<int(W)>(std::simd::cat(a, b), [](int j) { const int jc = j / 2; return (jc & 1) * int(W) + 2 * (jc / 2 + (Hi ? int(C) / 2 : 0)) + (j & 1); });
    }
    [[gnu::always_inline]] static V zipLo(const V& a, const V& b) { return zip<false>(a, b); }
    [[gnu::always_inline]] static V zipHi(const V& a, const V& b) { return zip<true>(a, b); }
#else
    template<bool Hi, std::size_t... J>
    [[gnu::always_inline]] static V zipImpl(const V& a, const V& b, std::index_sequence<J...>) {
        constexpr auto idx = [](std::size_t j) { const std::size_t jc = j / 2; return int((jc & 1) * W + 2 * (jc / 2 + (Hi ? C / 2 : 0)) + (j & 1)); };
        return fromVB(__builtin_shufflevector(toVB(a), toVB(b), idx(J)...));
    }
    [[gnu::always_inline]] static V zipLo(const V& a, const V& b) { return zipImpl<false>(a, b, std::make_index_sequence<W>{}); }
    [[gnu::always_inline]] static V zipHi(const V& a, const V& b) { return zipImpl<true>(a, b, std::make_index_sequence<W>{}); }
#endif

    // interleave 8 vectors s[k] (lane p) into out[8p + k] order, as 8 consecutive vectors
    [[gnu::always_inline]] static void interleave8(const V* s, V* out) {
        V r1[8], r2[8];
        // round 1: pairs (0,4) (2,6) (1,5) (3,7)
        constexpr int A[4] = {0, 2, 1, 3};
        for (int i = 0; i < 4; ++i) {
            r1[2 * i]     = zipLo(s[A[i]], s[A[i] + 4]);
            r1[2 * i + 1] = zipHi(s[A[i]], s[A[i] + 4]);
        }
        // round 2: streams u0=(r1[0],r1[1]) with u1=(r1[2],r1[3]); v0=(r1[4],r1[5]) with v1=(r1[6],r1[7])
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < 2; ++i) {
                r2[4 * h + 2 * i]     = zipLo(r1[4 * h + i], r1[4 * h + 2 + i]);
                r2[4 * h + 2 * i + 1] = zipHi(r1[4 * h + i], r1[4 * h + 2 + i]);
            }
        }
        // round 3: stream (r2[0..3]) with (r2[4..7])
        for (int i = 0; i < 4; ++i) {
            out[2 * i]     = zipLo(r2[i], r2[4 + i]);
            out[2 * i + 1] = zipHi(r2[i], r2[4 + i]);
        }
    }
};

struct AlignedFree {
    void operator()(void* p) const noexcept { std::free(p); }
};
template<typename T>
T* allocAligned(std::size_t n) {
    return static_cast<T*>(std::aligned_alloc(64, ((n * sizeof(T) + 63) / 64) * 64));
}

} // namespace detail

/// Forward FFT plan for one power-of-two size N >= 2. Plans are cheap to keep and not thread-safe (they own scratch).
template<std::floating_point T>
class FFT {
    using Ops = detail::Ops<T>;
    using V   = typename Ops::V;
    static constexpr std::size_t W = Ops::W, C = Ops::C;
public:
    struct Stage {
        std::size_t n, s, m, radix;
        std::size_t tw; // offset into _tw
    };

    FFT() = default;
    explicit FFT(std::size_t N) { init(N); }

    std::size_t size() const noexcept { return _N; }

    void init(std::size_t N) {
        if (N < 2 || !std::has_single_bit(N)) {
            throw std::invalid_argument("fastfft::FFT: size must be a power of two >= 2");
        }
        _N = N;
        _stages.clear();
        std::vector<T> tw;
        auto               w = [](std::size_t k, std::size_t nn) { const double a = -2.0 * std::numbers::pi * double(k) / double(nn); return std::complex<double>(std::cos(a), std::sin(a)); };
        _vector             = N >= 8 * 8 && (N / 8) % C == 0;
        if (_vector) {
            std::size_t n = N, s = 1;
            while (n > 1) {
                const std::size_t r = (s > 1 && n >= 16) ? 16 : ((n >= 8) ? 8 : n);
                const std::size_t m = n / r;
                _stages.push_back({n, s, m, r, tw.size()});
                if (s == 1) { // per vector of C p's: (r-1) x [W floats (wr,wr), W floats (-wi,wi)]
                    for (std::size_t v = 0; v < m / C; ++v) {
                        for (std::size_t k = 1; k < r; ++k) {
                            for (std::size_t j = 0; j < C; ++j) {
                                const auto z = w(k * (C * v + j), n);
                                tw.push_back(T(z.real()));
                                tw.push_back(T(z.real()));
                            }
                            for (std::size_t j = 0; j < C; ++j) {
                                const auto z = w(k * (C * v + j), n);
                                tw.push_back(T(-z.imag()));
                                tw.push_back(T(z.imag()));
                            }
                        }
                    }
                } else if (r >= 8) { // per p: (r-1) x [wr, wr, -wi, wi]
                    for (std::size_t p = 0; p < m; ++p) {
                        for (std::size_t k = 1; k < r; ++k) {
                            const auto z = w(k * p, n);
                            tw.insert(tw.end(), {T(z.real()), T(z.real()), T(-z.imag()), T(z.imag())});
                        }
                    }
                }
                n /= r;
                s *= r;
            }
        }
        _tw = std::move(tw);
        _work.reset(detail::allocAligned<T>(2 * N));
        _inCopy.reset(detail::allocAligned<T>(2 * N));
    }

    /// out[k] = sum_n in[n] exp(-2 pi i n k / N); in == out is allowed, no alignment requirement
    void forward(const std::complex<T>* inC, std::complex<T>* outC) const {
        if (!_vector) {
            scalarForward(inC, outC);
            return;
        }
        const T* in  = reinterpret_cast<const T*>(inC);
        T*       out = reinterpret_cast<T*>(outC);
        if (in == out) {
            std::memcpy(_inCopy.get(), in, 2 * _N * sizeof(T));
            in = _inCopy.get();
        }
        const std::size_t nS  = _stages.size();
        const T*      src = in;
        for (std::size_t i = 0; i < nS; ++i) {
            T* dst = ((nS - 1 - i) % 2 == 0) ? out : _work.get();
            runStage(_stages[i], src, dst);
            src = dst;
        }
    }

private:
    std::size_t                                 _N{0};
    bool                                        _vector{false};
    std::vector<Stage>                          _stages;
    std::vector<T>                          _tw;
    std::unique_ptr<T, detail::AlignedFree> _work, _inCopy;

    void runStage(const Stage& st, const T* x, T* y) const {
                const std::size_t m = st.m, s = st.s;
        if (st.radix == 2) {
            for (std::size_t q = 0; q < s; q += C) {
                const V a = Ops::load(x + 2 * q), b = Ops::load(x + 2 * (q + s));
                Ops::store(a + b, y + 2 * q);
                Ops::store(a - b, y + 2 * (q + s));
            }
            return;
        }
        if (st.radix == 4) {
            for (std::size_t q = 0; q < s; q += C) {
                V y0, y1, y2, y3;
                Ops::bfly4(Ops::load(x + 2 * q), Ops::load(x + 2 * (q + s)), Ops::load(x + 2 * (q + 2 * s)), Ops::load(x + 2 * (q + 3 * s)), y0, y1, y2, y3);
                Ops::store(y0, y + 2 * q);
                Ops::store(y1, y + 2 * (q + s));
                Ops::store(y2, y + 2 * (q + 2 * s));
                Ops::store(y3, y + 2 * (q + 3 * s));
            }
            return;
        }
        if (s == 1) { // radix 8, vectorised over p, interleaved on the store
            const T* t = _tw.data() + st.tw;
            for (std::size_t p0 = 0; p0 < m; p0 += C, t += 7 * 2 * W) {
                V a[8], o[8], r[8];
                for (std::size_t k = 0; k < 8; ++k) {
                    a[k] = Ops::load(x + 2 * (p0 + k * m));
                }
                Ops::bfly8(a, o);
                for (std::size_t k = 1; k < 8; ++k) {
                    o[k] = Ops::cmul(o[k], Ops::load(t + 2 * W * (k - 1)), Ops::load(t + 2 * W * (k - 1) + W));
                }
                Ops::interleave8(o, r);
                for (std::size_t i = 0; i < 8; ++i) {
                    Ops::store(r[i], y + 2 * 8 * p0 + W * i);
                }
            }
            return;
        }
        if (st.radix == 16) {
            for (std::size_t p = 0; p < m; ++p) {
                const T* t  = _tw.data() + st.tw + 60 * p;
                const T* xp = x + 2 * s * p;
                T*       yp = y + 2 * s * 16 * p;
                for (std::size_t q = 0; q < s; q += C) {
                    V a[16], o[16];
                    for (std::size_t k = 0; k < 16; ++k) {
                        a[k] = Ops::load(xp + 2 * (q + k * s * m));
                    }
                    Ops::bfly16(a, o);
                    Ops::store(o[0], yp + 2 * q);
                    for (std::size_t k = 1; k < 16; ++k) {
                        Ops::store(p == 0 ? o[k] : Ops::cmul(o[k], Ops::bcast2(t + 4 * (k - 1)), Ops::bcast2(t + 4 * (k - 1) + 2)), yp + 2 * (q + k * s));
                    }
                }
            }
            return;
        }
        for (std::size_t p = 0; p < m; ++p) { // radix 8
            const T* t  = _tw.data() + st.tw + 28 * p;
            const T* xp = x + 2 * s * p;
            T*       yp = y + 2 * s * 8 * p;
            for (std::size_t q = 0; q < s; q += C) {
                V a[8], o[8];
                for (std::size_t k = 0; k < 8; ++k) {
                    a[k] = Ops::load(xp + 2 * (q + k * s * m));
                }
                Ops::bfly8(a, o);
                Ops::store(o[0], yp + 2 * q);
                for (std::size_t k = 1; k < 8; ++k) {
                    Ops::store(p == 0 ? o[k] : Ops::cmul(o[k], Ops::bcast2(t + 4 * (k - 1)), Ops::bcast2(t + 4 * (k - 1) + 2)), yp + 2 * (q + k * s));
                }
            }
        }
    }

    void scalarForward(const std::complex<T>* in, std::complex<T>* out) const {
        const std::size_t                N = _N;
        std::vector<std::complex<T>> a(in, in + N), b(N);
        std::size_t                      n = N, s = 1;
        auto*                            x = a.data();
        auto*                            y = b.data();
        while (n > 1) {
            const std::size_t m = n / 2;
            for (std::size_t p = 0; p < m; ++p) {
                const double              ang = -2.0 * std::numbers::pi * double(p) / double(n);
                const std::complex<T> wp(T(std::cos(ang)), T(std::sin(ang)));
                for (std::size_t q = 0; q < s; ++q) {
                    const auto u           = x[q + s * p];
                    const auto v           = x[q + s * (p + m)];
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

} // namespace fastfft

#endif // FASTFFT_FFT_HPP
