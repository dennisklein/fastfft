// Faster replacement for the per-frame work of gr::blocks::fft::FFT<T, DataSet<float>> (spectrum mode):
// computeSpectrum() + assembleDataSet() fused into one call with the same output.
//  - window + FFT on aligned scratch with fastfft (no per-frame allocation, no buffered SimdFFT path)
//  - magnitude via sqrt(re^2 + im^2) and phase via a branchless atan2 (both auto-vectorised), written straight into
//    the DataSet at their fftshifted positions instead of computed, rotated and copied
//  - DataSet metadata (names, units, axis, meta maps) built once per settings change and copied per frame
#ifndef FASTFFT_SPECTRUM_HPP
#define FASTFFT_SPECTRUM_HPP

#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#include <gnuradio-4.0/fourier/fft.hpp>

#include "fastfft/FastFFT.hpp"

namespace fastfft {

namespace spectrum_detail {
// atan2 for float: octant reduction to [0, tan(pi/8)] + cephes atanf polynomial, branchless so loops vectorise.
// max error ~1.5e-7 rad vs std::atan2 (see test); signs/zeros follow std::atan2 except the NaN inputs.
[[gnu::always_inline]] inline float atan2f_fast(float y, float x) {
    const float ax = std::fabs(x), ay = std::fabs(y);
    const float mx = ax > ay ? ax : ay, mn = ax > ay ? ay : ax;
    float       t  = mx > 0.f ? mn / mx : 0.f; // in [0, 1]
    const bool  hi = t > 0.41421356237309503f;  // tan(pi/8)
    t              = hi ? (t - 1.f) / (t + 1.f) : t;
    const float z  = t * t;
    float       r  = ((((8.05374449538e-2f * z - 1.38776856032e-1f) * z + 1.99777106478e-1f) * z - 3.33329491539e-1f) * z) * t + t;
    r              = hi ? r + std::numbers::pi_v<float> / 4 : r;
    r              = ay > ax ? std::numbers::pi_v<float> / 2 - r : r;
    r              = std::signbit(x) ? std::numbers::pi_v<float> - r : r;
    return std::copysign(r, y);
}
} // namespace spectrum_detail

template<typename T> // T = std::complex<float> (full, fftshifted spectrum) or float (half spectrum)
class Spectrum {
public:
    using P                                 = float;
    using Block                             = gr::blocks::fft::FFT<T, gr::DataSet<P>>;
    static constexpr bool computeFullSpectrum = gr::meta::complex_like<T>;

    gr::DataSet<P> operator()(Block& blk, std::span<const T> in) {
        const std::size_t N = blk.fft_size;
        if (N != _N || blk.window_coefficients.size() != N || _templateKey != key(blk)) {
            rebuild(blk, N);
        }
        // window + FFT
        auto* x = reinterpret_cast<P*>(_in.get());
        for (std::size_t i = 0; i < N; ++i) {
            const P w = _window[i];
            if constexpr (computeFullSpectrum) {
                x[2 * i]     = in[i].real() * w;
                x[2 * i + 1] = in[i].imag() * w;
            } else {
                x[2 * i]     = in[i] * w;
                x[2 * i + 1] = 0;
            }
        }
        _plan.forward(_in.get(), _out.get());
        const P* X = reinterpret_cast<const P*>(_out.get());

        gr::DataSet<P> ds = _template; // metadata copy, signal_values already sized
        const std::size_t M     = _M;
        P*                mag   = ds.signal_values.data();
        P*                phase = mag + M;
        P*                re    = mag + 2 * M;
        P*                im    = mag + 3 * M;
        const P           scale2 = P(2) / P(N), scale1 = P(1) / P(N);
        // output bin j comes from FFT bin src(j): fftshifted for the full spectrum, identity for the half one
        const std::size_t shift = computeFullSpectrum ? N / 2 : 0;
        auto              fill  = [&](std::size_t j0, std::size_t j1, std::size_t s0) {
            for (std::size_t j = j0, s = s0; j < j1; ++j, ++s) {
                const P a = X[2 * s], b = X[2 * s + 1];
                re[j]     = a;
                im[j]     = b;
                mag[j]    = std::sqrt(a * a + b * b);
                phase[j]  = spectrum_detail::atan2f_fast(b, a);
            }
        };
        if constexpr (computeFullSpectrum) {
            fill(0, N - shift, shift); // negative frequencies first
            fill(N - shift, N, 0);
            for (std::size_t j = 0; j < M; ++j) {
                mag[j] *= scale2;
            }
        } else {
            fill(0, M, 0);
            for (std::size_t j = 0; j < M; ++j) {
                mag[j] *= (j == 0 || (N % 2 == 0 && j == N / 2)) ? scale1 : scale2;
            }
        }
        if (blk.output_in_db) {
            for (std::size_t j = 0; j < M; ++j) {
                mag[j] = mag[j] > P(0) ? P(20) * std::log10(mag[j]) : std::numeric_limits<P>::lowest();
            }
        }
        if (blk.unwrap_phase || blk.output_in_deg) { // rare path: redo phase in natural order like the original
            std::vector<P> nat(M);
            for (std::size_t j = 0; j < M; ++j) {
                nat[(j + shift) % (computeFullSpectrum ? N : M)] = phase[j];
            }
            if (blk.unwrap_phase) {
                gr::algorithm::fft::unwrapPhase(std::span<P>(nat));
            }
            if (blk.output_in_deg) {
                for (auto& v : nat) {
                    v = gr::algorithm::fft::radToDeg(v);
                }
            }
            for (std::size_t j = 0; j < M; ++j) {
                phase[j] = nat[(j + shift) % (computeFullSpectrum ? N : M)];
            }
        }
        for (std::size_t sIdx = 0; sIdx < 4; ++sIdx) { // single pass min/max per signal (vectorises)
            const P* v  = mag + sIdx * M;
            P        lo = v[0], hi = v[0];
            for (std::size_t j = 1; j < M; ++j) {
                lo = std::min(lo, v[j]);
                hi = std::max(hi, v[j]);
            }
            ds.signal_ranges[sIdx] = {lo, hi};
        }
        return ds;
    }

private:
    std::size_t                                    _N{0}, _M{0};
    Plan                                           _plan;
    std::unique_ptr<std::complex<P>[], void (*)(std::complex<P>*)> _in{nullptr, nullptr}, _out{nullptr, nullptr};
    std::vector<P>                                 _window;
    gr::DataSet<P>                                 _template;
    std::string                                    _templateKey;

    static std::string key(const Block& b) {
        return std::format("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}", static_cast<std::size_t>(b.fft_size), static_cast<float>(b.sample_rate), b.signal_name.value, b.signal_unit.value, static_cast<bool>(b.output_in_db), static_cast<bool>(b.output_in_deg), static_cast<bool>(b.unwrap_phase), static_cast<int>(b.window.value), b.input_chunk_size, b.output_chunk_size, b.stride);
    }

    static std::complex<P>* allocC(std::size_t n) { return static_cast<std::complex<P>*>(std::aligned_alloc(64, ((n * sizeof(std::complex<P>) + 63) / 64) * 64)); }
    static void             freeC(std::complex<P>* p) { std::free(p); }

    void rebuild(Block& blk, std::size_t N) {
        blk.ensureHostScratch(N); // window coefficients as the block itself builds them
        _N = N;
        _M = computeFullSpectrum ? N : N / 2 + 1;
        _plan.init(N);
        _in  = {allocC(N), &freeC};
        _out = {allocC(N), &freeC};
        _window.assign(blk.window_coefficients.begin(), blk.window_coefficients.end());
        // build the metadata exactly as the block does, by letting it assemble once from zeroed spectra
        std::ranges::fill(blk._magnitudeSpectrum, P(0));
        std::ranges::fill(blk._phaseSpectrum, P(0));
        std::ranges::fill(blk._outData, std::complex<P>(0));
        _template     = blk.assembleDataSet();
        _templateKey  = key(blk);
    }
};

} // namespace fastfft

#endif
