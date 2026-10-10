// spectrum mode: gr4 FFT<T, DataSet<float>> computeSpectrum()+assembleDataSet() vs. fastfft::Spectrum, same input,
// outputs compared field by field
#include <benchmark.hpp>

#include <complex>
#include <format>
#include <print>
#include <random>
#include <vector>

#include "FastSpectrum.hpp"

using C = std::complex<float>;

namespace {
template<typename T>
double maxRelDiff(std::span<const float> a, std::span<const float> b) {
    float scale = 0;
    for (float v : a) scale = std::max(scale, std::abs(v));
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, double(std::abs(a[i] - b[i])));
    return scale > 0 ? d / scale : d;
}

template<typename T>
void run(std::size_t N, bool flags = false) {
    using Block = gr::blocks::fft::FFT<T, gr::DataSet<float>>;
    auto blk    = std::make_shared<Block>(gr::property_map{{"fft_size", static_cast<gr::Size_t>(N)}});
    blk->init(blk->progress);
    if (flags) {
        blk->output_in_db  = true;
        blk->output_in_deg = true;
        blk->unwrap_phase  = true;
    }
    std::mt19937                          gen(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto                                  in = std::make_shared<std::vector<T>>(N);
    for (auto& v : *in) {
        if constexpr (std::is_same_v<T, C>) v = {dist(gen), dist(gen)};
        else v = dist(gen);
    }
    auto fast = std::make_shared<fastfft::Spectrum<T>>();

    // verification
    blk->computeSpectrum(std::span<const T>(*in));
    const auto ref = blk->assembleDataSet();
    const auto got = (*fast)(*blk, std::span<const T>(*in));
    const std::size_t M = ref.signalValues(0).size();
    double dMag = maxRelDiff<T>(ref.signalValues(0), got.signalValues(0));
    double dRe = maxRelDiff<T>(ref.signalValues(2), got.signalValues(2)), dIm = maxRelDiff<T>(ref.signalValues(3), got.signalValues(3));
    double dPh = 0; // absolute radians, ignoring +-pi wrap on near-real bins and bins with negligible magnitude
    std::size_t phCount = 0;
    for (std::size_t j = 0; j < M; ++j) {
        double d = std::abs(ref.signalValues(1)[j] - got.signalValues(1)[j]);
        d        = std::min(d, std::abs(d - 2 * std::numbers::pi));
        if (flags || ref.signalValues(0)[j] > 1e-4f * ref.signalRange(0).max) { dPh = std::max(dPh, d); ++phCount; }
    }
    const bool metaSame = ref.signal_names == got.signal_names && ref.signal_units == got.signal_units && ref.axis_values == got.axis_values && ref.meta_information == got.meta_information && ref.extents == got.extents;
    if (flags) { // dB magnitude and unwrapped degrees: compare absolutely (dB, deg)
        double a = 0, b = 0;
        for (std::size_t j = 0; j < M; ++j) {
            a = std::max(a, double(std::abs(ref.signalValues(0)[j] - got.signalValues(0)[j])));
            b = std::max(b, double(std::abs(ref.signalValues(1)[j] - got.signalValues(1)[j])));
        }
        const bool metaSame = ref.meta_information == got.meta_information;
        std::println("{:>7} N={:>6} [dB, deg, unwrap]: |dB diff| {:.1e}  |deg diff| {:.1e}  metadata {}", std::is_same_v<T, C> ? "complex" : "real", N, a, b, metaSame ? "identical" : "DIFFERS");
        boost::ut::expect(a < 1e-3 && b < 1e-1 && metaSame);
        return;
    }
    std::println("{:>7} N={:>6}: mag {:.1e}  phase {:.1e} rad  re {:.1e}  im {:.1e}  metadata {}", std::is_same_v<T, C> ? "complex" : "real", N, dMag, dPh, dRe, dIm, metaSame ? "identical" : "DIFFERS");
    boost::ut::expect(dMag < 1e-5 && dRe < 1e-5 && dIm < 1e-5 && dPh < 1e-4 && metaSame);

    const std::string tn = std::is_same_v<T, C> ? "complex" : "real";
    ::benchmark::benchmark<200>(std::format("{:>7} N={:>6} gr4  computeSpectrum+assembleDataSet", tn, N)) = [blk, in] {
        blk->computeSpectrum(std::span<const T>(*in));
        auto ds = blk->assembleDataSet();
        return ds.signal_values.size();
    };
    ::benchmark::benchmark<200>(std::format("{:>7} N={:>6} fast Spectrum", tn, N)) = [blk, in, fast] {
        auto ds = (*fast)(*blk, std::span<const T>(*in));
        return ds.signal_values.size();
    };
    ::benchmark::results::add_separator();
}
} // namespace

const boost::ut::suite<"FFT spectrum mode: gr4 vs fast"> benchmarks = [] {
    for (std::size_t N : {1024UZ, 4096UZ, 65536UZ}) run<C>(N);
    for (std::size_t N : {1024UZ, 4096UZ, 65536UZ}) run<float>(N);
    run<C>(4096, true);
    run<float>(1024, true);
};

int main() {}
