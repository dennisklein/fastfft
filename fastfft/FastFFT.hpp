// Hand-written AVX-512 complex<float> FFT for power-of-two sizes (interleaved complex, out-of-place or in-place).
// N < 64: radix-4 Stockham; 64 <= N < 2^20: radix-8 first stage + radix-16 Stockham stages; N >= 2^20: four-step.
#ifndef FASTFFT_HPP
#define FASTFFT_HPP

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <span>
#include <vector>

namespace fastfft {

using V = __m512;

namespace detail {

struct AlignedFree {
    void operator()(float* p) const noexcept { std::free(p); }
};

inline float* allocFloats(std::size_t n) { return static_cast<float*>(std::aligned_alloc(64, ((n * sizeof(float) + 63) / 64) * 64 + 64)); }

[[gnu::always_inline]] inline V swapReIm(V z) { return _mm512_permute_ps(z, 0xB1); }

// z * w with w given as duplicated real/imag parts
[[gnu::always_inline]] inline V cmul(V z, V wr, V wi) { return _mm512_fmaddsub_ps(z, wr, _mm512_mul_ps(swapReIm(z), wi)); }

[[gnu::always_inline]] inline void bfly4(V a, V b, V c, V d, V& y0, V& y1, V& y2, V& y3) {
    const V one = _mm512_set1_ps(1.f);
    const V apc = _mm512_add_ps(a, c);
    const V amc = _mm512_sub_ps(a, c);
    const V bpd = _mm512_add_ps(b, d);
    const V sw  = swapReIm(_mm512_sub_ps(b, d));
    y0          = _mm512_add_ps(apc, bpd);
    y2          = _mm512_sub_ps(apc, bpd);
    y1          = _mm512_fmsubadd_ps(amc, one, sw); // amc - j(b-d)
    y3          = _mm512_fmaddsub_ps(amc, one, sw); // amc + j(b-d)
}


[[gnu::always_inline]] inline V negJ(V v) { // -j * v
    return _mm512_xor_ps(swapReIm(v), _mm512_castsi512_ps(_mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL))));
}

[[gnu::always_inline]] inline V mulW8(V v) { // v * exp(-j pi/4)
    const V r = _mm512_set1_ps(0.70710678118654752f);
    return _mm512_fmsubadd_ps(v, r, _mm512_mul_ps(swapReIm(v), r));
}

[[gnu::always_inline]] inline V mulW8_3(V v) { // v * exp(-3j pi/4)
    const V nr = _mm512_set1_ps(-0.70710678118654752f);
    return _mm512_fmaddsub_ps(v, nr, _mm512_mul_ps(swapReIm(v), nr));
}

// 8-point DFT, natural order in and out
[[gnu::always_inline]] inline void bfly8(const V* a, V* y) {
    const V b0 = _mm512_add_ps(a[0], a[4]), b1 = _mm512_add_ps(a[1], a[5]), b2 = _mm512_add_ps(a[2], a[6]), b3 = _mm512_add_ps(a[3], a[7]);
    const V c0 = _mm512_sub_ps(a[0], a[4]);
    const V c1 = mulW8(_mm512_sub_ps(a[1], a[5]));
    const V c2 = negJ(_mm512_sub_ps(a[2], a[6]));
    const V c3 = mulW8_3(_mm512_sub_ps(a[3], a[7]));
    bfly4(b0, b1, b2, b3, y[0], y[2], y[4], y[6]);
    bfly4(c0, c1, c2, c3, y[1], y[3], y[5], y[7]);
}

// 16-point DFT (4x4), natural order in and out
[[gnu::always_inline]] inline void bfly16(const V* a, V* y) {
    const V c1r = _mm512_set1_ps(0.92387953251128674f), c1i = _mm512_set1_ps(-0.38268343236508977f); // W16^1
    const V c3r = _mm512_set1_ps(0.38268343236508977f), c3i = _mm512_set1_ps(-0.92387953251128674f); // W16^3
    const V c9r = _mm512_set1_ps(-0.92387953251128674f), c9i = _mm512_set1_ps(0.38268343236508977f); // W16^9
    V B[4][4];
    for (int j1 = 0; j1 < 4; ++j1) {
        bfly4(a[j1], a[j1 + 4], a[j1 + 8], a[j1 + 12], B[j1][0], B[j1][1], B[j1][2], B[j1][3]);
    }
    B[1][1] = cmul(B[1][1], c1r, c1i);
    B[1][2] = mulW8(B[1][2]);
    B[1][3] = cmul(B[1][3], c3r, c3i);
    B[2][1] = mulW8(B[2][1]);
    B[2][2] = negJ(B[2][2]);
    B[2][3] = mulW8_3(B[2][3]);
    B[3][1] = cmul(B[3][1], c3r, c3i);
    B[3][2] = mulW8_3(B[3][2]);
    B[3][3] = cmul(B[3][3], c9r, c9i);
    for (int k = 0; k < 4; ++k) {
        bfly4(B[0][k], B[1][k], B[2][k], B[3][k], y[k], y[k + 4], y[k + 8], y[k + 12]);
    }
}

// 8x8 transpose of 64-bit (complex<float>) elements
[[gnu::always_inline]] inline void transpose8x8(V* r) {
    __m512d t[8], u[8];
    for (int i = 0; i < 4; ++i) {
        t[2 * i]     = _mm512_unpacklo_pd(_mm512_castps_pd(r[2 * i]), _mm512_castps_pd(r[2 * i + 1]));
        t[2 * i + 1] = _mm512_unpackhi_pd(_mm512_castps_pd(r[2 * i]), _mm512_castps_pd(r[2 * i + 1]));
    }
    for (int h = 0; h < 2; ++h) {
        const __m512d* tt = t + 4 * h;
        u[4 * h + 0]      = _mm512_shuffle_f64x2(tt[0], tt[2], 0x88);
        u[4 * h + 1]      = _mm512_shuffle_f64x2(tt[0], tt[2], 0xDD);
        u[4 * h + 2]      = _mm512_shuffle_f64x2(tt[1], tt[3], 0x88);
        u[4 * h + 3]      = _mm512_shuffle_f64x2(tt[1], tt[3], 0xDD);
    }
    r[0] = _mm512_castpd_ps(_mm512_shuffle_f64x2(u[0], u[4], 0x88));
    r[4] = _mm512_castpd_ps(_mm512_shuffle_f64x2(u[0], u[4], 0xDD));
    r[2] = _mm512_castpd_ps(_mm512_shuffle_f64x2(u[1], u[5], 0x88));
    r[6] = _mm512_castpd_ps(_mm512_shuffle_f64x2(u[1], u[5], 0xDD));
    r[1] = _mm512_castpd_ps(_mm512_shuffle_f64x2(u[2], u[6], 0x88));
    r[5] = _mm512_castpd_ps(_mm512_shuffle_f64x2(u[2], u[6], 0xDD));
    r[3] = _mm512_castpd_ps(_mm512_shuffle_f64x2(u[3], u[7], 0x88));
    r[7] = _mm512_castpd_ps(_mm512_shuffle_f64x2(u[3], u[7], 0xDD));
}

} // namespace detail

class StockhamPlan {
public:
    enum class Mode { Lane1, Lane4, Bcast, Radix2 };
    struct Stage {
        Mode        mode;
        std::size_t n, s, m;
        float*      tw; // layout depends on mode
    };

    StockhamPlan() = default;
    explicit StockhamPlan(std::size_t N) { init(N); }
    StockhamPlan(const StockhamPlan&)            = delete;
    StockhamPlan& operator=(const StockhamPlan&) = delete;
    ~StockhamPlan() { release(); }

    std::size_t size() const noexcept { return _N; }

    void init(std::size_t N) {
        release();
        _N = N;
        if (N < 32) {
            return;
        }
        std::size_t n = N, s = 1;
        while (n > 1) {
            if (n == 2) {
                _stages.push_back({Mode::Radix2, n, s, 1, nullptr});
                break;
            }
            const std::size_t m = n / 4;
            Stage             st{s == 1 ? Mode::Lane1 : (s == 4 ? Mode::Lane4 : Mode::Bcast), n, s, m, nullptr};
            auto              w = [n](std::size_t k) { const double a = -2.0 * std::numbers::pi * double(k) / double(n); return std::complex<double>(std::cos(a), std::sin(a)); };
            if (st.mode == Mode::Bcast) {
                st.tw = detail::allocFloats(6 * m);
                for (std::size_t p = 0; p < m; ++p) {
                    for (std::size_t k = 1; k < 4; ++k) {
                        auto z                     = w(k * p);
                        st.tw[6 * p + 2 * (k - 1)] = float(z.real());
                        st.tw[6 * p + 2 * (k - 1) + 1] = float(z.imag());
                    }
                }
            } else {
                // per vector (8 complex lanes): [R1 I1 R2 I2 R3 I3] each 16 floats
                const std::size_t nVec = (st.mode == Mode::Lane1) ? m / 8 : m / 2;
                st.tw                  = detail::allocFloats(96 * nVec);
                for (std::size_t v = 0; v < nVec; ++v) {
                    for (std::size_t j = 0; j < 8; ++j) {
                        const std::size_t p = (st.mode == Mode::Lane1) ? 8 * v + j : 2 * v + j / 4;
                        for (std::size_t k = 1; k < 4; ++k) {
                            auto   z = w(k * p);
                            float* R = st.tw + 96 * v + 32 * (k - 1);
                            R[2 * j] = R[2 * j + 1] = float(z.real());
                            R[16 + 2 * j] = R[16 + 2 * j + 1] = float(z.imag());
                        }
                    }
                }
            }
            _stages.push_back(st);
            n /= 4;
            s *= 4;
        }
        _work   = detail::allocFloats(2 * N);
        _inCopy = detail::allocFloats(2 * N);
    }

    // out-of-place or in-place forward transform, X[k] = sum x[n] exp(-2 pi i n k / N)
    void forward(const std::complex<float>* inC, std::complex<float>* outC) const {
        const std::size_t N = _N;
        if (N < 32) {
            scalarForward(inC, outC);
            return;
        }
        const float* in  = reinterpret_cast<const float*>(inC);
        float*       out = reinterpret_cast<float*>(outC);
        if (in == out) {
            std::memcpy(_inCopy, in, 2 * N * sizeof(float));
            in = _inCopy;
        }
        const std::size_t nS  = _stages.size();
        const float*      src = in;
        for (std::size_t i = 0; i < nS; ++i) {
            float* dst = ((nS - 1 - i) % 2 == 0) ? out : _work;
            runStage(_stages[i], src, dst);
            src = dst;
        }
        if (src != out) {
            std::memcpy(out, src, 2 * N * sizeof(float));
        }
    }

private:
    std::size_t        _N{0};
    std::vector<Stage> _stages;
    float*             _work{nullptr};
    float*             _inCopy{nullptr};

    void release() {
        for (auto& st : _stages) {
            std::free(st.tw);
        }
        _stages.clear();
        std::free(_work);
        std::free(_inCopy);
        _work   = nullptr;
        _inCopy = nullptr;
    }

    static void runStage(const Stage& st, const float* x, float* y) {
        using namespace detail;
        const std::size_t m = st.m, s = st.s;
        switch (st.mode) {
        case Mode::Radix2: {
            for (std::size_t q = 0; q < s; q += 8) {
                const V a = _mm512_loadu_ps(x + 2 * q);
                const V b = _mm512_loadu_ps(x + 2 * (q + s));
                _mm512_storeu_ps(y + 2 * q, _mm512_add_ps(a, b));
                _mm512_storeu_ps(y + 2 * (q + s), _mm512_sub_ps(a, b));
            }
        } break;
        case Mode::Bcast: {
            for (std::size_t p = 0; p < m; ++p) {
                const float* t   = st.tw + 6 * p;
                const V      w1r = _mm512_set1_ps(t[0]), w1i = _mm512_set1_ps(t[1]);
                const V      w2r = _mm512_set1_ps(t[2]), w2i = _mm512_set1_ps(t[3]);
                const V      w3r = _mm512_set1_ps(t[4]), w3i = _mm512_set1_ps(t[5]);
                const float* xp  = x + 2 * s * p;
                float*       yp  = y + 2 * s * 4 * p;
                for (std::size_t q = 0; q < s; q += 8) {
                    V y0, y1, y2, y3;
                    bfly4(_mm512_loadu_ps(xp + 2 * q), _mm512_loadu_ps(xp + 2 * (q + s * m)), _mm512_loadu_ps(xp + 2 * (q + 2 * s * m)), _mm512_loadu_ps(xp + 2 * (q + 3 * s * m)), y0, y1, y2, y3);
                    _mm512_storeu_ps(yp + 2 * q, y0);
                    _mm512_storeu_ps(yp + 2 * (q + s), cmul(y1, w1r, w1i));
                    _mm512_storeu_ps(yp + 2 * (q + 2 * s), cmul(y2, w2r, w2i));
                    _mm512_storeu_ps(yp + 2 * (q + 3 * s), cmul(y3, w3r, w3i));
                }
            }
        } break;
        case Mode::Lane1: {
            for (std::size_t p0 = 0, v = 0; p0 < m; p0 += 8, ++v) {
                const float* t = st.tw + 96 * v;
                V            y0, y1, y2, y3;
                bfly4(_mm512_loadu_ps(x + 2 * p0), _mm512_loadu_ps(x + 2 * (p0 + m)), _mm512_loadu_ps(x + 2 * (p0 + 2 * m)), _mm512_loadu_ps(x + 2 * (p0 + 3 * m)), y0, y1, y2, y3);
                y1 = cmul(y1, _mm512_load_ps(t), _mm512_load_ps(t + 16));
                y2 = cmul(y2, _mm512_load_ps(t + 32), _mm512_load_ps(t + 48));
                y3 = cmul(y3, _mm512_load_ps(t + 64), _mm512_load_ps(t + 80));
                // transpose 4x8 complex (64-bit elements) -> y[4p + k]
                const __m512d a = _mm512_castps_pd(y0), b = _mm512_castps_pd(y1), c = _mm512_castps_pd(y2), d = _mm512_castps_pd(y3);
                const __m512d t0 = _mm512_unpacklo_pd(a, b), t1 = _mm512_unpackhi_pd(a, b);
                const __m512d t2 = _mm512_unpacklo_pd(c, d), t3 = _mm512_unpackhi_pd(c, d);
                // lanes L0..L3 of t0 = (a_{2i}, b_{2i}); of t1 = (a_{2i+1}, b_{2i+1})
                const __m512d u0 = _mm512_shuffle_f64x2(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)); // t0.L0 t0.L1 t2.L0 t2.L1
                const __m512d u1 = _mm512_shuffle_f64x2(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)); // t1.L0 t1.L1 t3.L0 t3.L1
                const __m512d u2 = _mm512_shuffle_f64x2(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
                const __m512d u3 = _mm512_shuffle_f64x2(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
                float*        yp = y + 2 * 4 * p0;
                _mm512_storeu_pd(reinterpret_cast<double*>(yp), _mm512_shuffle_f64x2(u0, u1, _MM_SHUFFLE(2, 0, 2, 0)));      // t0.L0 t2.L0 t1.L0 t3.L0
                _mm512_storeu_pd(reinterpret_cast<double*>(yp + 16), _mm512_shuffle_f64x2(u0, u1, _MM_SHUFFLE(3, 1, 3, 1))); // t0.L1 t2.L1 t1.L1 t3.L1
                _mm512_storeu_pd(reinterpret_cast<double*>(yp + 32), _mm512_shuffle_f64x2(u2, u3, _MM_SHUFFLE(2, 0, 2, 0)));
                _mm512_storeu_pd(reinterpret_cast<double*>(yp + 48), _mm512_shuffle_f64x2(u2, u3, _MM_SHUFFLE(3, 1, 3, 1)));
            }
        } break;
        case Mode::Lane4: {
            for (std::size_t p0 = 0, v = 0; p0 < m; p0 += 2, ++v) {
                const float* t  = st.tw + 96 * v;
                const float* xp = x + 2 * 4 * p0;
                V            y0, y1, y2, y3;
                bfly4(_mm512_loadu_ps(xp), _mm512_loadu_ps(xp + 2 * 4 * m), _mm512_loadu_ps(xp + 2 * 8 * m), _mm512_loadu_ps(xp + 2 * 12 * m), y0, y1, y2, y3);
                y1        = cmul(y1, _mm512_load_ps(t), _mm512_load_ps(t + 16));
                y2        = cmul(y2, _mm512_load_ps(t + 32), _mm512_load_ps(t + 48));
                y3        = cmul(y3, _mm512_load_ps(t + 64), _mm512_load_ps(t + 80));
                float* yp = y + 2 * 16 * p0;
                _mm512_storeu_ps(yp, _mm512_shuffle_f32x4(y0, y1, _MM_SHUFFLE(1, 0, 1, 0)));
                _mm512_storeu_ps(yp + 16, _mm512_shuffle_f32x4(y2, y3, _MM_SHUFFLE(1, 0, 1, 0)));
                _mm512_storeu_ps(yp + 32, _mm512_shuffle_f32x4(y0, y1, _MM_SHUFFLE(3, 2, 3, 2)));
                _mm512_storeu_ps(yp + 48, _mm512_shuffle_f32x4(y2, y3, _MM_SHUFFLE(3, 2, 3, 2)));
            }
        } break;
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
                    const auto u         = x[q + s * p];
                    const auto v         = x[q + s * (p + m)];
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

/// L-point FFTs on arrays of vectors: every SIMD lane (8 complex) is an independent transform, so no shuffles and
/// only broadcast twiddles. Radix-8 Stockham with one radix-2/-4 stage last when log2(L) is not a multiple of 3.
/// Element j of the input is the vector at `x + j * xs` (floats), element k of the output the vector at `y + k * ys`.
class VecFFT {
public:
    struct Stage {
        std::size_t n, s, m, radix;
        std::size_t tw; // offset into _tw (2 * (radix-1) floats per p)
    };

    void init(std::size_t L) {
        _L = L;
        _stages.clear();
        std::vector<float> tw;
        std::size_t        n = L, s = 1;
        while (n > 1) {
            const std::size_t r = (n >= 8) ? 8 : n; // 8, then a final 4 or 2 (m == 1, no twiddles)
            const std::size_t m = n / r;
            _stages.push_back({n, s, m, r, tw.size()});
            for (std::size_t p = 0; p < m; ++p) {
                for (std::size_t k = 1; k < r; ++k) {
                    const double a = -2.0 * std::numbers::pi * double(k * p) / double(n);
                    tw.push_back(float(std::cos(a)));
                    tw.push_back(float(std::sin(a)));
                }
            }
            n /= r;
            s *= r;
        }
        _tw = std::move(tw);
        _buf.reset(detail::allocFloats(2 * 16 * L));
    }

    std::size_t size() const noexcept { return _L; }

    void run(const float* x, std::size_t xs, float* y, std::size_t ys) const {
        const std::size_t nS   = _stages.size();
        float*            bufA = _buf.get();
        float*            bufB = _buf.get() + 16 * _L;
        const float*      src  = x;
        std::size_t       ss   = xs;
        for (std::size_t i = 0; i < nS; ++i) {
            float*      dst = (i + 1 == nS) ? y : ((i % 2 == 0) ? bufA : bufB);
            std::size_t ds  = (i + 1 == nS) ? ys : 16;
            stage(_stages[i], src, ss, dst, ds);
            src = dst;
            ss  = ds;
        }
    }

private:
    std::size_t                                 _L{0};
    std::vector<Stage>                          _stages;
    std::vector<float>                          _tw;
    std::unique_ptr<float, detail::AlignedFree> _buf;

    void stage(const Stage& st, const float* x, std::size_t xs, float* y, std::size_t ys) const {
        using namespace detail;
        const std::size_t m = st.m, s = st.s;
        if (st.radix == 2) {
            for (std::size_t q = 0; q < s; ++q) {
                const V a = _mm512_load_ps(x + q * xs);
                const V b = _mm512_load_ps(x + (q + s) * xs);
                _mm512_store_ps(y + q * ys, _mm512_add_ps(a, b));
                _mm512_store_ps(y + (q + s) * ys, _mm512_sub_ps(a, b));
            }
            return;
        }
        if (st.radix == 4) {
            for (std::size_t q = 0; q < s; ++q) {
                V y0, y1, y2, y3;
                bfly4(_mm512_load_ps(x + q * xs), _mm512_load_ps(x + (q + s) * xs), _mm512_load_ps(x + (q + 2 * s) * xs), _mm512_load_ps(x + (q + 3 * s) * xs), y0, y1, y2, y3);
                _mm512_store_ps(y + q * ys, y0);
                _mm512_store_ps(y + (q + s) * ys, y1);
                _mm512_store_ps(y + (q + 2 * s) * ys, y2);
                _mm512_store_ps(y + (q + 3 * s) * ys, y3);
            }
            return;
        }
        for (std::size_t p = 0; p < m; ++p) {
            const float* t = _tw.data() + st.tw + 14 * p;
            for (std::size_t q = 0; q < s; ++q) {
                V a[8], o[8];
                for (std::size_t k = 0; k < 8; ++k) {
                    a[k] = _mm512_load_ps(x + (q + s * (p + k * m)) * xs);
                }
                bfly8(a, o);
                _mm512_store_ps(y + (q + s * (8 * p)) * ys, o[0]);
                for (std::size_t k = 1; k < 8; ++k) {
                    _mm512_store_ps(y + (q + s * (8 * p + k)) * ys, p == 0 ? o[k] : cmul(o[k], _mm512_set1_ps(t[2 * (k - 1)]), _mm512_set1_ps(t[2 * (k - 1) + 1])));
                }
            }
        }
    }
};

/// radix-8 Stockham on contiguous complex data, N >= 64: first stage (stride 1) vectorised over p with an 8x8
/// transpose on the store, later stages vectorised over q (stride >= 8) with broadcast twiddles
class Stockham8Plan {
public:
    struct Stage {
        std::size_t n, s, m, radix;
        float*      tw;
    };
    static inline bool useRadix16 = true; // tunable
    Stockham8Plan() = default;
    explicit Stockham8Plan(std::size_t N) { init(N); }
    Stockham8Plan(const Stockham8Plan&)            = delete;
    Stockham8Plan& operator=(const Stockham8Plan&) = delete;
    ~Stockham8Plan() { release(); }

    std::size_t size() const noexcept { return _N; }

    void init(std::size_t N) {
        release();
        _N            = N;
        std::size_t n = N, s = 1;
        auto        w = [](std::size_t k, std::size_t nn) { const double a = -2.0 * std::numbers::pi * double(k) / double(nn); return std::complex<double>(std::cos(a), std::sin(a)); };
        while (n > 1) {
            const std::size_t r = (s > 1 && useRadix16 && n >= 16) ? 16 : ((n >= 8) ? 8 : n);
            const std::size_t m = n / r;
            Stage             st{n, s, m, r, nullptr};
            if (s == 1) { // per-lane twiddles: per vector of 8 p's, 7 x (16 re-dup, 16 im-dup)
                st.tw = detail::allocFloats(224 * (m / 8));
                for (std::size_t v = 0; v < m / 8; ++v) {
                    for (std::size_t j = 0; j < 8; ++j) {
                        for (std::size_t k = 1; k < 8; ++k) {
                            const auto z = w(k * (8 * v + j), n);
                            float*     R = st.tw + 224 * v + 32 * (k - 1);
                            R[2 * j] = R[2 * j + 1] = float(z.real());
                            R[16 + 2 * j] = R[16 + 2 * j + 1] = float(z.imag());
                        }
                    }
                }
            } else if (r >= 8) {
                st.tw = detail::allocFloats(2 * (r - 1) * m);
                for (std::size_t p = 0; p < m; ++p) {
                    for (std::size_t k = 1; k < r; ++k) {
                        const auto z                          = w(k * p, n);
                        st.tw[2 * (r - 1) * p + 2 * (k - 1)]     = float(z.real());
                        st.tw[2 * (r - 1) * p + 2 * (k - 1) + 1] = float(z.imag());
                    }
                }
            }
            _stages.push_back(st);
            n /= r;
            s *= r;
        }
        _work   = detail::allocFloats(2 * N);
        _inCopy = detail::allocFloats(2 * N);
    }

    void forward(const std::complex<float>* inC, std::complex<float>* outC) const {
        const float* in  = reinterpret_cast<const float*>(inC);
        float*       out = reinterpret_cast<float*>(outC);
        if (in == out) {
            std::memcpy(_inCopy, in, 2 * _N * sizeof(float));
            in = _inCopy;
        }
        const std::size_t nS  = _stages.size();
        const float*      src = in;
        for (std::size_t i = 0; i < nS; ++i) {
            float* dst = ((nS - 1 - i) % 2 == 0) ? out : _work;
            runStage(_stages[i], src, dst);
            src = dst;
        }
    }

private:
    std::size_t        _N{0};
    std::vector<Stage> _stages;
    float*             _work{nullptr};
    float*             _inCopy{nullptr};

    void release() {
        for (auto& st : _stages) {
            std::free(st.tw);
        }
        _stages.clear();
        std::free(_work);
        std::free(_inCopy);
        _work   = nullptr;
        _inCopy = nullptr;
    }

    static void runStage(const Stage& st, const float* x, float* y) {
        using namespace detail;
        const std::size_t m = st.m, s = st.s;
        if (st.radix == 2) {
            for (std::size_t q = 0; q < s; q += 8) {
                const V a = _mm512_loadu_ps(x + 2 * q);
                const V b = _mm512_loadu_ps(x + 2 * (q + s));
                _mm512_storeu_ps(y + 2 * q, _mm512_add_ps(a, b));
                _mm512_storeu_ps(y + 2 * (q + s), _mm512_sub_ps(a, b));
            }
            return;
        }
        if (st.radix == 4) {
            for (std::size_t q = 0; q < s; q += 8) {
                V y0, y1, y2, y3;
                bfly4(_mm512_loadu_ps(x + 2 * q), _mm512_loadu_ps(x + 2 * (q + s)), _mm512_loadu_ps(x + 2 * (q + 2 * s)), _mm512_loadu_ps(x + 2 * (q + 3 * s)), y0, y1, y2, y3);
                _mm512_storeu_ps(y + 2 * q, y0);
                _mm512_storeu_ps(y + 2 * (q + s), y1);
                _mm512_storeu_ps(y + 2 * (q + 2 * s), y2);
                _mm512_storeu_ps(y + 2 * (q + 3 * s), y3);
            }
            return;
        }
        if (s == 1) {
            for (std::size_t p0 = 0, v = 0; p0 < m; p0 += 8, ++v) {
                const float* t = st.tw + 224 * v;
                V            a[8], o[8];
                for (std::size_t k = 0; k < 8; ++k) {
                    a[k] = _mm512_loadu_ps(x + 2 * (p0 + k * m));
                }
                bfly8(a, o);
                for (std::size_t k = 1; k < 8; ++k) {
                    o[k] = cmul(o[k], _mm512_load_ps(t + 32 * (k - 1)), _mm512_load_ps(t + 32 * (k - 1) + 16));
                }
                transpose8x8(o);
                for (std::size_t i = 0; i < 8; ++i) {
                    _mm512_storeu_ps(y + 2 * 8 * (p0 + i), o[i]);
                }
            }
            return;
        }
        if (st.radix == 16) {
            for (std::size_t p = 0; p < m; ++p) {
                const float* t  = st.tw + 30 * p;
                const float* xp = x + 2 * s * p;
                float*       yp = y + 2 * s * 16 * p;
                for (std::size_t q = 0; q < s; q += 8) {
                    V a[16], o[16];
                    for (std::size_t k = 0; k < 16; ++k) {
                        a[k] = _mm512_loadu_ps(xp + 2 * (q + k * s * m));
                    }
                    bfly16(a, o);
                    _mm512_storeu_ps(yp + 2 * q, o[0]);
                    for (std::size_t k = 1; k < 16; ++k) {
                        _mm512_storeu_ps(yp + 2 * (q + k * s), p == 0 ? o[k] : cmul(o[k], _mm512_set1_ps(t[2 * (k - 1)]), _mm512_set1_ps(t[2 * (k - 1) + 1])));
                    }
                }
            }
            return;
        }
        for (std::size_t p = 0; p < m; ++p) {
            const float* t  = st.tw + 14 * p;
            const float* xp = x + 2 * s * p;
            float*       yp = y + 2 * s * 8 * p;
            for (std::size_t q = 0; q < s; q += 8) {
                V a[8], o[8];
                for (std::size_t k = 0; k < 8; ++k) {
                    a[k] = _mm512_loadu_ps(xp + 2 * (q + k * s * m));
                }
                bfly8(a, o);
                _mm512_storeu_ps(yp + 2 * q, o[0]);
                for (std::size_t k = 1; k < 8; ++k) {
                    _mm512_storeu_ps(yp + 2 * (q + k * s), cmul(o[k], _mm512_set1_ps(t[2 * (k - 1)]), _mm512_set1_ps(t[2 * (k - 1) + 1])));
                }
            }
        }
    }
};

/// four-step FFT: N = N1 * N2, (A) N1 column FFTs of size N2 (8 columns per SIMD vector), twiddle W_N^(n1 k2),
/// 8x8 transposes into a scratch matrix, (B) N2 FFTs of size N1 (8 per vector) written straight to the output.
class FourStepPlan {
public:
    static inline int  n2Bias = 0; // log2(N2) = ceil(log2(N)/2) + n2Bias, tunable
    static inline bool padT   = true;

    void init(std::size_t N) {
        _N          = N;
        const int l = std::countr_zero(N);
        const int l2 = std::clamp((l + 1) / 2 + n2Bias, 3, l - 3);
        _N2         = std::size_t{1} << l2;
        _N1         = N / _N2;
        _TS         = padT ? _N2 + 8 : _N2; // padded row stride breaks the power-of-two cache-set aliasing
        _colA.init(_N2);
        _rowB.init(_N1);
        _colBuf.reset(detail::allocFloats(2 * 8 * _N2));
        _T.reset(detail::allocFloats(2 * _N1 * _TS));
        _tw.reset(detail::allocFloats(4 * N)); // per (c, k2): 16 floats re-dup, 16 floats im-dup
        for (std::size_t c = 0; c < _N1 / 8; ++c) {
            for (std::size_t k2 = 0; k2 < _N2; ++k2) {
                float* R = _tw.get() + 32 * (c * _N2 + k2);
                for (std::size_t i = 0; i < 8; ++i) {
                    const double a = -2.0 * std::numbers::pi * double((8 * c + i) * k2) / double(N);
                    R[2 * i] = R[2 * i + 1] = float(std::cos(a));
                    R[16 + 2 * i] = R[16 + 2 * i + 1] = float(std::sin(a));
                }
            }
        }
    }

    std::size_t size() const noexcept { return _N; }

    void forward(const float* x, float* X) const {
        using namespace detail;
        const std::size_t N1 = _N1, N2 = _N2;
        float*            T  = _T.get();
        for (std::size_t c = 0; c < N1 / 8; ++c) { // (A)
            _colA.run(x + 16 * c, 2 * N1, _colBuf.get(), 16);
            const float* tw = _tw.get() + 32 * c * N2;
            for (std::size_t r = 0; r < N2 / 8; ++r) {
                V v[8];
                for (std::size_t j = 0; j < 8; ++j) {
                    const std::size_t k2 = 8 * r + j;
                    v[j]                 = cmul(_mm512_load_ps(_colBuf.get() + 16 * k2), _mm512_load_ps(tw + 32 * k2), _mm512_load_ps(tw + 32 * k2 + 16));
                }
                transpose8x8(v);
                for (std::size_t i = 0; i < 8; ++i) {
                    _mm512_store_ps(T + 2 * ((8 * c + i) * _TS + 8 * r), v[i]);
                }
            }
        }
        for (std::size_t r = 0; r < N2 / 8; ++r) { // (B)
            _rowB.run(T + 16 * r, 2 * _TS, X + 16 * r, 2 * N2);
        }
    }

private:
    std::size_t                                 _N{0}, _N1{0}, _N2{0}, _TS{0};
    VecFFT                                      _colA, _rowB;
    std::unique_ptr<float, detail::AlignedFree> _colBuf, _T, _tw;
};

/// dispatcher: picks the variant per size (thresholds measured on the AVX-512 host, see README)
class Plan {
public:
    Plan() = default;
    explicit Plan(std::size_t N) { init(N); }

    static inline std::size_t fourStepMin = std::size_t{1} << 20; // tunable

    void init(std::size_t N) {
        _N    = N;
        _kind = N < 64 ? Kind::Radix4 : (N < fourStepMin ? Kind::Stockham : Kind::FourStep);
        switch (_kind) {
        case Kind::Radix4: _r4.init(N); break;
        case Kind::Stockham: _st.init(N); break;
        case Kind::FourStep:
            _fs.init(N);
            _tmpIn.reset(detail::allocFloats(2 * N));
            _tmpOut.reset(detail::allocFloats(2 * N));
            break;
        }
    }
    std::size_t size() const noexcept { return _N; }

    void forward(const std::complex<float>* in, std::complex<float>* out) const {
        switch (_kind) {
        case Kind::Radix4: _r4.forward(in, out); return;
        case Kind::Stockham: _st.forward(in, out); return;
        case Kind::FourStep: break;
        }
        const float* x = reinterpret_cast<const float*>(in);
        float*       y = reinterpret_cast<float*>(out);
        if (x == y || (reinterpret_cast<std::uintptr_t>(x) & 63U) != 0) { // aliasing / unaligned input: stage a copy
            std::memcpy(_tmpIn.get(), x, 2 * _N * sizeof(float));
            x = _tmpIn.get();
        }
        if ((reinterpret_cast<std::uintptr_t>(y) & 63U) != 0) { // unaligned output: transform into scratch, then copy
            _fs.forward(x, _tmpOut.get());
            std::memcpy(y, _tmpOut.get(), 2 * _N * sizeof(float));
            return;
        }
        _fs.forward(x, y);
    }

private:
    enum class Kind { Radix4, Stockham, FourStep };
    std::size_t                                 _N{0};
    Kind                                        _kind{Kind::Radix4};
    StockhamPlan                                _r4;
    Stockham8Plan                               _st;
    FourStepPlan                                _fs;
    std::unique_ptr<float, detail::AlignedFree> _tmpIn, _tmpOut;
};

} // namespace fastfft

#endif
