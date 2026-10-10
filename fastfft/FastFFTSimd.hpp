// Width-generic port of FastFFT.hpp (radix-8 first stage + radix-16 Stockham) written against std(::experimental)::simd
// via vir::stdx::native_simd,
// the std::simd flavour gnuradio4 uses. No intrinsics: the vector width follows the target (16 floats on AVX-512,
// 8 on AVX2, 4 on SSE/NEON). complex<float>, power-of-two N.
#ifndef FASTFFT_SIMD_HPP
#define FASTFFT_SIMD_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

#if defined(FASTFFT_USE_STD_EXPERIMENTAL_SIMD) // libstdc++'s implementation directly (also usable from clang)
#include <experimental/simd>
namespace fastfft::stdsimd {
namespace stdx = std::experimental;
}
#else // vir-simd, as gnuradio4 uses: libstdc++'s implementation on GCC, vir's own fallback on clang
#include <vir/simd.h>
namespace fastfft::stdsimd {
namespace stdx = vir::stdx;
}
#endif

namespace fastfft::stdsimd {

#ifdef FASTFFT_SIMD_FLOATS // force a width (floats per vector), e.g. 4/8/16 to compare widths on one ISA
using V  = stdx::simd<float, stdx::simd_abi::deduce_t<float, FASTFFT_SIMD_FLOATS>>;
using VD = stdx::simd<double, stdx::simd_abi::deduce_t<double, FASTFFT_SIMD_FLOATS / 2>>;
#else
using V  = stdx::native_simd<float>;
using VD = stdx::native_simd<double>;
#endif

inline constexpr std::size_t W = V::size(); // floats per vector
inline constexpr std::size_t C = W / 2;     // complex values per vector
static_assert(C >= 2 && C <= 8, "needs 2..8 complex lanes per vector");

namespace detail {

// a * b + c; spelled out (with -ffp-contract=fast this is one FMA) because stdx::fma is not inlined by libstdc++
[[gnu::always_inline]] inline V fma(const V& a, const V& b, const V& c) { return a * b + c; }

[[gnu::always_inline]] inline V load(const float* p) { return V(p, stdx::element_aligned); }
[[gnu::always_inline]] inline void store(const V& v, float* p) { v.copy_to(p, stdx::element_aligned); }

[[gnu::always_inline]] inline V pm() { return V([](auto i) { return (i & 1) ? 1.f : -1.f; }); } // (-1, +1, ...)
[[gnu::always_inline]] inline V mp() { return V([](auto i) { return (i & 1) ? -1.f : 1.f; }); } // (+1, -1, ...)

typedef float VB __attribute__((vector_size(sizeof(float) * W))); // the compiler's generic vector type of the same width
static_assert(sizeof(VB) == sizeof(V));

template<std::size_t... J>
[[gnu::always_inline]] inline V swapImpl(const V& v, std::index_sequence<J...>) {
    const VB x = std::bit_cast<VB>(v);
    return std::bit_cast<V>(__builtin_shufflevector(x, x, int(J ^ 1U)...));
}
[[gnu::always_inline]] inline V swapReIm(const V& v) { return swapImpl(v, std::make_index_sequence<W>{}); }

// broadcast one 64-bit pair of floats to every complex lane
[[gnu::always_inline]] inline V bcast2(const float* p) {
    double d;
    std::memcpy(&d, p, sizeof(d));
    return std::bit_cast<V>(VD(d));
}

// z * w, with w given as (wr, wr) and (-wi, wi) per lane
[[gnu::always_inline]] inline V cmul(const V& z, const V& wr, const V& wiS) { return fma(z, wr, swapReIm(z) * wiS); }

[[gnu::always_inline]] inline void bfly4(const V& a, const V& b, const V& c, const V& d, V& y0, V& y1, V& y2, V& y3) {
    const V apc = a + c, amc = a - c, bpd = b + d;
    const V sw  = swapReIm(b - d);
    y0          = apc + bpd;
    y2          = apc - bpd;
    y1          = fma(sw, mp(), amc); // amc - j(b-d)
    y3          = fma(sw, pm(), amc); // amc + j(b-d)
}

[[gnu::always_inline]] inline V negJ(const V& v) { return swapReIm(v) * mp(); }
[[gnu::always_inline]] inline V mulW8(const V& v) {
    constexpr float r = 0.70710678118654752f;
    return fma(swapReIm(v), mp() * r, v * r);
}
[[gnu::always_inline]] inline V mulW8_3(const V& v) {
    constexpr float r = 0.70710678118654752f;
    return fma(swapReIm(v), mp() * r, v * -r);
}

[[gnu::always_inline]] inline void bfly8(const V* a, V* y) {
    const V b0 = a[0] + a[4], b1 = a[1] + a[5], b2 = a[2] + a[6], b3 = a[3] + a[7];
    const V c0 = a[0] - a[4], c1 = mulW8(a[1] - a[5]), c2 = negJ(a[2] - a[6]), c3 = mulW8_3(a[3] - a[7]);
    bfly4(b0, b1, b2, b3, y[0], y[2], y[4], y[6]);
    bfly4(c0, c1, c2, c3, y[1], y[3], y[5], y[7]);
}

[[gnu::always_inline]] inline V cconst(float re) { return V(re); }
[[gnu::always_inline]] inline V cconstS(float im) { return pm() * im; }

[[gnu::always_inline]] inline void bfly16(const V* a, V* y) {
    constexpr float c1 = 0.92387953251128674f, s1 = 0.38268343236508977f;
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
// std::simd has no two-input shuffle; permuting `concat(a, b)` gets lowered to element moves (~3x the cost of the whole
// FFT stage), so this one helper drops to the compiler's generic vector extension, the same mechanism vir's
// simd_permute uses internally. Still no ISA intrinsics and still width-generic.
template<bool Hi, std::size_t... J>
[[gnu::always_inline]] inline V zipImpl(const V& a, const V& b, std::index_sequence<J...>) {
    constexpr auto idx = [](std::size_t j) { const std::size_t jc = j / 2; return int((jc & 1) * W + 2 * (jc / 2 + (Hi ? C / 2 : 0)) + (j & 1)); };
    return std::bit_cast<V>(__builtin_shufflevector(std::bit_cast<VB>(a), std::bit_cast<VB>(b), idx(J)...));
}
[[gnu::always_inline]] inline V zipLo(const V& a, const V& b) { return zipImpl<false>(a, b, std::make_index_sequence<W>{}); }
[[gnu::always_inline]] inline V zipHi(const V& a, const V& b) { return zipImpl<true>(a, b, std::make_index_sequence<W>{}); }

// interleave 8 vectors s[k] (lane p) into out[8p + k] order, as 8 consecutive vectors
[[gnu::always_inline]] inline void interleave8(const V* s, V* out) {
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

struct AlignedFree {
    void operator()(float* p) const noexcept { std::free(p); }
};
inline float* allocFloats(std::size_t n) { return static_cast<float*>(std::aligned_alloc(64, ((n * sizeof(float) + 63) / 64) * 64 + 64)); }

} // namespace detail

class Plan {
public:
    struct Stage {
        std::size_t n, s, m, radix;
        std::size_t tw; // offset into _tw
    };

    Plan() = default;
    explicit Plan(std::size_t N) { init(N); }

    std::size_t size() const noexcept { return _N; }

    void init(std::size_t N) {
        _N = N;
        _stages.clear();
        std::vector<float> tw;
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
                                tw.push_back(float(z.real()));
                                tw.push_back(float(z.real()));
                            }
                            for (std::size_t j = 0; j < C; ++j) {
                                const auto z = w(k * (C * v + j), n);
                                tw.push_back(float(-z.imag()));
                                tw.push_back(float(z.imag()));
                            }
                        }
                    }
                } else if (r >= 8) { // per p: (r-1) x [wr, wr, -wi, wi]
                    for (std::size_t p = 0; p < m; ++p) {
                        for (std::size_t k = 1; k < r; ++k) {
                            const auto z = w(k * p, n);
                            tw.insert(tw.end(), {float(z.real()), float(z.real()), float(-z.imag()), float(z.imag())});
                        }
                    }
                }
                n /= r;
                s *= r;
            }
        }
        _tw = std::move(tw);
        _work.reset(detail::allocFloats(2 * N));
        _inCopy.reset(detail::allocFloats(2 * N));
    }

    void forward(const std::complex<float>* inC, std::complex<float>* outC) const {
        if (!_vector) {
            scalarForward(inC, outC);
            return;
        }
        const float* in  = reinterpret_cast<const float*>(inC);
        float*       out = reinterpret_cast<float*>(outC);
        if (in == out) {
            std::memcpy(_inCopy.get(), in, 2 * _N * sizeof(float));
            in = _inCopy.get();
        }
        const std::size_t nS  = _stages.size();
        const float*      src = in;
        for (std::size_t i = 0; i < nS; ++i) {
            float* dst = ((nS - 1 - i) % 2 == 0) ? out : _work.get();
            runStage(_stages[i], src, dst);
            src = dst;
        }
    }

private:
    std::size_t                                 _N{0};
    bool                                        _vector{false};
    std::vector<Stage>                          _stages;
    std::vector<float>                          _tw;
    std::unique_ptr<float, detail::AlignedFree> _work, _inCopy;

    void runStage(const Stage& st, const float* x, float* y) const {
        using namespace detail;
        const std::size_t m = st.m, s = st.s;
        if (st.radix == 2) {
            for (std::size_t q = 0; q < s; q += C) {
                const V a = load(x + 2 * q), b = load(x + 2 * (q + s));
                store(a + b, y + 2 * q);
                store(a - b, y + 2 * (q + s));
            }
            return;
        }
        if (st.radix == 4) {
            for (std::size_t q = 0; q < s; q += C) {
                V y0, y1, y2, y3;
                bfly4(load(x + 2 * q), load(x + 2 * (q + s)), load(x + 2 * (q + 2 * s)), load(x + 2 * (q + 3 * s)), y0, y1, y2, y3);
                store(y0, y + 2 * q);
                store(y1, y + 2 * (q + s));
                store(y2, y + 2 * (q + 2 * s));
                store(y3, y + 2 * (q + 3 * s));
            }
            return;
        }
        if (s == 1) { // radix 8, vectorised over p, interleaved on the store
            const float* t = _tw.data() + st.tw;
            for (std::size_t p0 = 0; p0 < m; p0 += C, t += 7 * 2 * W) {
                V a[8], o[8], r[8];
                for (std::size_t k = 0; k < 8; ++k) {
                    a[k] = load(x + 2 * (p0 + k * m));
                }
                bfly8(a, o);
                for (std::size_t k = 1; k < 8; ++k) {
                    o[k] = cmul(o[k], load(t + 2 * W * (k - 1)), load(t + 2 * W * (k - 1) + W));
                }
                interleave8(o, r);
                for (std::size_t i = 0; i < 8; ++i) {
                    store(r[i], y + 2 * 8 * p0 + W * i);
                }
            }
            return;
        }
        if (st.radix == 16) {
            for (std::size_t p = 0; p < m; ++p) {
                const float* t  = _tw.data() + st.tw + 60 * p;
                const float* xp = x + 2 * s * p;
                float*       yp = y + 2 * s * 16 * p;
                for (std::size_t q = 0; q < s; q += C) {
                    V a[16], o[16];
                    for (std::size_t k = 0; k < 16; ++k) {
                        a[k] = load(xp + 2 * (q + k * s * m));
                    }
                    bfly16(a, o);
                    store(o[0], yp + 2 * q);
                    for (std::size_t k = 1; k < 16; ++k) {
                        store(p == 0 ? o[k] : cmul(o[k], bcast2(t + 4 * (k - 1)), bcast2(t + 4 * (k - 1) + 2)), yp + 2 * (q + k * s));
                    }
                }
            }
            return;
        }
        for (std::size_t p = 0; p < m; ++p) { // radix 8
            const float* t  = _tw.data() + st.tw + 28 * p;
            const float* xp = x + 2 * s * p;
            float*       yp = y + 2 * s * 8 * p;
            for (std::size_t q = 0; q < s; q += C) {
                V a[8], o[8];
                for (std::size_t k = 0; k < 8; ++k) {
                    a[k] = load(xp + 2 * (q + k * s * m));
                }
                bfly8(a, o);
                store(o[0], yp + 2 * q);
                for (std::size_t k = 1; k < 8; ++k) {
                    store(p == 0 ? o[k] : cmul(o[k], bcast2(t + 4 * (k - 1)), bcast2(t + 4 * (k - 1) + 2)), yp + 2 * (q + k * s));
                }
            }
        }
    }

    void scalarForward(const std::complex<float>* in, std::complex<float>* out) const {
        const std::size_t                N = _N;
        std::vector<std::complex<float>> a(in, in + N), b(N);
        std::size_t                      n = N, s = 1;
        auto*                            x = a.data();
        auto*                            y = b.data();
        while (n > 1) {
            const std::size_t m = n / 2;
            for (std::size_t p = 0; p < m; ++p) {
                const double              ang = -2.0 * std::numbers::pi * double(p) / double(n);
                const std::complex<float> wp(float(std::cos(ang)), float(std::sin(ang)));
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

} // namespace fastfft::stdsimd

#endif
