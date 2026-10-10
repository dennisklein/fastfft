// where the time goes in the spectrum (DataSet) mode of gr::blocks::fft::FFT: full processBulk vs. its parts
#include <benchmark.hpp>

#include <complex>
#include <format>
#include <random>
#include <vector>

#include <gnuradio-4.0/fourier/fft.hpp>

#include "fastfft/FastFFT.hpp"

using C = std::complex<float>;

namespace {
template<typename T>
void run(std::size_t N) {
    using Block = gr::blocks::fft::FFT<T, gr::DataSet<float>>;
    auto blk    = std::make_shared<Block>(gr::property_map{{"fft_size", static_cast<gr::Size_t>(N)}});
    blk->init(blk->progress);
    std::mt19937                          gen(1);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto                                  in = std::make_shared<std::vector<T>>(N);
    for (auto& v : *in) {
        if constexpr (std::is_same_v<T, C>) {
            v = {dist(gen), dist(gen)};
        } else {
            v = dist(gen);
        }
    }
    blk->computeSpectrum(std::span<const T>(*in)); // warm up / allocate
    const std::string tn = std::is_same_v<T, C> ? "complex" : "real";
    ::benchmark::benchmark<200>(std::format("{:>7} N={:>6} computeSpectrum+assembleDataSet", tn, N)) = [blk, in] {
        blk->computeSpectrum(std::span<const T>(*in));
        auto ds = blk->assembleDataSet();
        return ds.signal_values.size();
    };
    ::benchmark::benchmark<200>(std::format("{:>7} N={:>6} computeSpectrum (window+FFT+mag+phase)", tn, N)) = [blk, in] { blk->computeSpectrum(std::span<const T>(*in)); };
    ::benchmark::benchmark<200>(std::format("{:>7} N={:>6} assembleDataSet", tn, N)) = [blk] { auto ds = blk->assembleDataSet(); return ds.signal_values.size(); };
    if constexpr (std::is_same_v<T, C>) {
        auto out  = std::make_shared<std::vector<C, gr::allocator::Aligned<C>>>(N);
        auto ina  = std::make_shared<std::vector<C, gr::allocator::Aligned<C>>>(in->begin(), in->end());
        auto algo = std::make_shared<gr::algorithm::FFT<C, C>>();
        auto plan = std::make_shared<fastfft::Plan>(N);
        ::benchmark::benchmark<200>(std::format("{:>7} N={:>6} FFT only (SimdFFT)", tn, N)) = [algo, ina, out] { algo->compute(std::span<const C>(*ina), std::span<C>(*out)); };
        ::benchmark::benchmark<200>(std::format("{:>7} N={:>6} FFT only (fastfft)", tn, N)) = [plan, ina, out] { plan->forward(ina->data(), out->data()); };
    }
    ::benchmark::results::add_separator();
}
} // namespace

const boost::ut::suite<"FFT spectrum mode breakdown"> benchmarks = [] {
    for (std::size_t N : {1024UZ, 4096UZ, 65536UZ}) {
        run<C>(N);
    }
    for (std::size_t N : {1024UZ, 65536UZ}) {
        run<float>(N);
    }
};

int main() {}
