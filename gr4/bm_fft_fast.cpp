// Minimal benchmark: gr4 `FFT<float>` streaming block (SimdFFT) vs. `FastFFT<float>` (hand-written AVX-512),
// using the gr4 benchmark facility. Also checks that both produce the same output on the same input.
#include <benchmark.hpp>

#include <bit>
#include <complex>
#include <format>
#include <numbers>
#include <print>
#include <random>
#include <vector>

#include <gnuradio-4.0/fourier/fft.hpp>

#include "fastfft/FastFFTBlock.hpp" // copied next to this file by scripts/build_gr4.sh

using C = std::complex<float>;

namespace {

constexpr std::size_t kSamplesPerCall = 1UZ << 16; // each timed call processes this many samples (>= one frame)

template<typename TBlock>
struct Harness { // ports wired once and reused, as in bm_fft_block
    gr::PortOut<C> srcOut;
    gr::PortIn<C>  fftIn{};
    gr::PortOut<C> fftOut;
    gr::PortIn<C>  sinkIn{};
    TBlock         block;
    std::size_t    total;

    Harness(std::size_t N, std::size_t total_) : block({{"fft_size", static_cast<gr::Size_t>(N)}}), total(total_) {
        block.init(block.progress);
        boost::ut::expect(srcOut.resizeBuffer(total).has_value());
        boost::ut::expect(fftOut.resizeBuffer(total).has_value());
        boost::ut::expect(srcOut.connect(fftIn).has_value());
        boost::ut::expect(fftOut.connect(sinkIn).has_value());
    }

    template<typename Fn>
    void run(const std::vector<C>& signal, Fn&& onOutput) {
        {
            auto w = srcOut.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(total);
            std::ranges::copy(signal, w.begin());
            w.publish(total);
        }
        auto inSpan = fftIn.template get<gr::SpanReleasePolicy::ProcessAll>(total);
        {
            auto outSpan = fftOut.template tryReserve<gr::SpanReleasePolicy::ProcessAll>(total);
            if (block.processBulk(inSpan, outSpan) != gr::work::Status::OK) {
                throw std::runtime_error("processBulk failed");
            }
        }
        auto readBack = sinkIn.template get<gr::SpanReleasePolicy::ProcessAll>(total);
        onOutput(std::span<const C>(readBack.data(), readBack.size()));
    }
};

std::vector<C> randomSignal(std::size_t n, unsigned seed) {
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<C>                        v(n);
    for (auto& z : v) {
        z = {dist(gen), dist(gen)};
    }
    return v;
}

std::vector<std::complex<double>> referenceDFT(std::span<const C> x) { // double-precision recursive radix-2
    std::vector<std::complex<double>> a(x.begin(), x.end());
    auto rec = [](auto& self, std::vector<std::complex<double>>& v) -> void {
        const std::size_t n = v.size();
        if (n == 1) {
            return;
        }
        std::vector<std::complex<double>> e(n / 2), o(n / 2);
        for (std::size_t i = 0; i < n / 2; ++i) {
            e[i] = v[2 * i];
            o[i] = v[2 * i + 1];
        }
        self(self, e);
        self(self, o);
        for (std::size_t k = 0; k < n / 2; ++k) {
            const auto t = std::polar(1.0, -2.0 * std::numbers::pi * double(k) / double(n)) * o[k];
            v[k]         = e[k] + t;
            v[k + n / 2] = e[k] - t;
        }
    };
    rec(rec, a);
    return a;
}

void runBenchmarks() {
    using namespace gr::blocks::fft;

    std::println("{:>8} {:>8} | {:>14} {:>14} {:>14}", "N", "frames", "maxdiff/max", "err(gr)/max", "err(fast)/max");
    for (std::size_t N : {64UZ, 256UZ, 1024UZ, 4096UZ, 16384UZ, 65536UZ, 262144UZ, 1048576UZ}) {
        const std::size_t total    = std::max(N, kSamplesPerCall);
        const std::size_t nBatches = total / N;
        const auto        signal   = randomSignal(total, 42U + static_cast<unsigned>(N));

        auto ref  = std::make_shared<Harness<FFT<float>>>(N, total);
        auto fast = std::make_shared<Harness<FastFFT<float>>>(N, total);

        // --- verification: identical inputs -> outputs agree (and both agree with a double-precision reference)
        std::vector<C> outRef, outFast;
        ref->run(signal, [&](auto s) { outRef.assign(s.begin(), s.end()); });
        fast->run(signal, [&](auto s) { outFast.assign(s.begin(), s.end()); });
        double maxDiff = 0, maxMag = 0, errRef = 0, errFast = 0;
        for (std::size_t b = 0; b < std::min<std::size_t>(nBatches, 4); ++b) {
            const auto exact = referenceDFT(std::span<const C>(signal).subspan(b * N, N));
            for (std::size_t i = 0; i < N; ++i) {
                maxMag  = std::max(maxMag, std::abs(exact[i]));
                errRef  = std::max(errRef, std::abs(std::complex<double>(outRef[b * N + i]) - exact[i]));
                errFast = std::max(errFast, std::abs(std::complex<double>(outFast[b * N + i]) - exact[i]));
            }
        }
        for (std::size_t i = 0; i < total; ++i) {
            maxDiff = std::max(maxDiff, double(std::abs(outRef[i] - outFast[i])));
        }
        std::println("{:>8} {:>8} | {:>14.3e} {:>14.3e} {:>14.3e}", N, nBatches, maxDiff / maxMag, errRef / maxMag, errFast / maxMag);
        boost::ut::expect(errFast / maxMag < 1e-5) << "FastFFT accuracy, N=" << N;
        boost::ut::expect(maxDiff / maxMag < 1e-5) << "FastFFT vs FFT, N=" << N;

        // --- timing, FLOP count per FFTW convention: 5 N log2(N) per transform
        const std::size_t flops = nBatches * 5UZ * N * static_cast<std::size_t>(std::countr_zero(N));
        const auto sig = std::make_shared<std::vector<C>>(signal);
        ::benchmark::benchmark<200>(std::format("block FFT<float>     (SimdFFT) N={:>7}", N), flops) = [ref, sig] { ref->run(*sig, [](auto) {}); };
        ::benchmark::benchmark<200>(std::format("block FastFFT<float> (AVX-512) N={:>7}", N), flops) = [fast, sig] { fast->run(*sig, [](auto) {}); };

        // kernel-only timing (no port copy): same aligned buffers, same frames
        auto inAligned  = std::make_shared<std::vector<C, gr::allocator::Aligned<C>>>(signal.begin(), signal.end());
        auto outAligned = std::make_shared<std::vector<C, gr::allocator::Aligned<C>>>(total);
        auto algo       = std::make_shared<gr::algorithm::FFT<C, C>>();
        auto plan       = std::make_shared<fastfft::Plan>(N);
        ::benchmark::benchmark<200>(std::format("kernel SimdFFT            N={:>7}", N), flops) = [=] {
            for (std::size_t b = 0; b < nBatches; ++b) {
                algo->compute(std::span<const C>(inAligned->data() + b * N, N), std::span<C>(outAligned->data() + b * N, N));
            }
        };
        ::benchmark::benchmark<200>(std::format("kernel fastfft            N={:>7}", N), flops) = [=] {
            for (std::size_t b = 0; b < nBatches; ++b) {
                plan->forward(inAligned->data() + b * N, outAligned->data() + b * N);
            }
        };
        ::benchmark::results::add_separator();
    }
}

} // namespace

const boost::ut::suite<"FastFFT benchmarks"> benchmarks = [] { runBenchmarks(); };

int main() { /* the UT runner executes the suites after main returns */ }
