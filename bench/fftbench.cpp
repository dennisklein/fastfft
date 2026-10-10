// fftbench -- timing and accuracy of complex forward FFTs (power-of-two sizes), one implementation per row.
//
// Method (see README): for each (precision, size) every implementation is first checked against a long-double
// reference DFT, then calibrated so that one timing sample lasts >= --min-sample-ms. One warm-up sample is discarded.
// Then --samples rounds are taken; in every round the implementations are measured in a freshly shuffled order, so
// slow drifts (thermal, frequency, neighbours) spread evenly over all of them and paired (same-round) ratios can be
// formed. A sample transforms max(N, 2^16) complex values as consecutive frames of size N (streaming, cache-warm),
// and is reported as nanoseconds per N-point transform. Raw samples go to CSV; statistics are done by analyze.py.
#include <fastfft/fft.hpp>
#ifdef FFTBENCH_WITH_CX
#include <fastfft/fft_cx.hpp> // C++26 std::simd::vec<std::complex<T>> variant
#endif

#ifdef FFTBENCH_WITH_SIMDFFT
#include <gnuradio-4.0/algorithm/fourier/SimdFFT.hpp>
// SimdFFT only needs these two out-of-line members on its error paths; defining them avoids linking gnuradio-core
gr::exception::exception(std::string_view, std::source_location) noexcept {}
const char* gr::exception::what() const noexcept { return "gr::exception"; }
#endif
#ifdef FFTBENCH_WITH_FFTW
#include <fftw3.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    int         samples     = 40;
    double      minSampleMs = 5.0;
    std::size_t minLog2 = 6, maxLog2 = 20;
    bool        doFloat = true, doDouble = true;
    std::string out     = "raw.csv";
    std::string label   = "";
    std::string impls   = ""; // comma-separated filter, empty = all
    bool        check   = false;
    bool        failed  = false;
};

struct Impl {
    std::string           name;
    std::function<void()> batch; // transforms all frames once
    long                  reps = 1;
};

template<typename T>
struct Buffers {
    std::size_t     N, frames;
    std::complex<T>*x, *y;
    Buffers(std::size_t n, std::size_t total) : N(n), frames(total / n) {
        x = static_cast<std::complex<T>*>(std::aligned_alloc(64, total * sizeof(std::complex<T>)));
        y = static_cast<std::complex<T>*>(std::aligned_alloc(64, total * sizeof(std::complex<T>)));
        fill();
    }
    void fill() { // uniform random input in [-1, 1)^2, the same for every implementation
        std::mt19937                      gen(static_cast<unsigned>(N));
        std::uniform_real_distribution<T> d(-1, 1);
        for (std::size_t i = 0; i < N * frames; ++i) {
            x[i] = {d(gen), d(gen)};
        }
    }
    ~Buffers() {
        std::free(x);
        std::free(y);
    }
};

// long-double radix-2 reference, recursive; returns max |X - ref| / max |ref| over the first frame
template<typename T>
double relError(const std::complex<T>* in, const std::complex<T>* out, std::size_t N) {
    using L = std::complex<long double>;
    std::function<void(std::vector<L>&)> rec = [&](std::vector<L>& a) {
        const std::size_t n = a.size();
        if (n == 1) {
            return;
        }
        std::vector<L> e(n / 2), o(n / 2);
        for (std::size_t i = 0; i < n / 2; ++i) {
            e[i] = a[2 * i];
            o[i] = a[2 * i + 1];
        }
        rec(e);
        rec(o);
        for (std::size_t k = 0; k < n / 2; ++k) {
            const L t    = std::polar(1.0L, -2.0L * std::numbers::pi_v<long double> * static_cast<long double>(k) / static_cast<long double>(n)) * o[k];
            a[k]         = e[k] + t;
            a[k + n / 2] = e[k] - t;
        }
    };
    std::vector<L> r(in, in + N);
    rec(r);
    long double e = 0, m = 0;
    for (std::size_t i = 0; i < N; ++i) {
        e = std::max(e, std::abs(L(out[i]) - r[i]));
        m = std::max(m, std::abs(r[i]));
    }
    return static_cast<double>(e / m);
}

double timeOnce(const Impl& im) {
    const auto t0 = Clock::now();
    for (long r = 0; r < im.reps; ++r) {
        im.batch();
    }
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

template<typename T>
void runSize(Options& opt, std::size_t N, std::FILE* csv, std::FILE* acc, std::mt19937& shuffleRng) {
    const char* prec = sizeof(T) == 4 ? "float" : "double";
    Buffers<T>  buf(N, std::max<std::size_t>(N, std::size_t{1} << 16));
    std::vector<Impl> impls;

    auto fast = std::make_shared<fastfft::FFT<T>>(N);
    impls.push_back({"fastfft", [fast, &buf] {
                         for (std::size_t b = 0; b < buf.frames; ++b) {
                             fast->forward(buf.x + b * buf.N, buf.y + b * buf.N);
                         }
                     }});
#ifdef FFTBENCH_WITH_CX
    auto cxf = std::make_shared<fastfft::cx::FFT<T>>(N);
    impls.push_back({"fastfft-cx", [cxf, &buf] {
                         for (std::size_t b = 0; b < buf.frames; ++b) {
                             cxf->forward(buf.x + b * buf.N, buf.y + b * buf.N);
                         }
                     }});
#endif
#ifdef FFTBENCH_WITH_SIMDFFT
    if (N >= 16) {
        using namespace gr::algorithm;
        auto s = std::make_shared<SimdFFT<T, Transform::Complex>>();
        s->resize(N);
        impls.push_back({"gr4-SimdFFT", [s, &buf] {
                             for (std::size_t b = 0; b < buf.frames; ++b) {
                                 s->template transform<Direction::Forward, Order::Ordered>(std::span<const T>(reinterpret_cast<const T*>(buf.x + b * buf.N), 2 * buf.N), std::span<T>(reinterpret_cast<T*>(buf.y + b * buf.N), 2 * buf.N));
                             }
                         }});
    }
#endif
#ifdef FFTBENCH_WITH_FFTW
    std::shared_ptr<void> fftwPlan;
    if constexpr (std::is_same_v<T, float>) {
        auto* p  = fftwf_plan_dft_1d(int(N), reinterpret_cast<fftwf_complex*>(buf.x), reinterpret_cast<fftwf_complex*>(buf.y), FFTW_FORWARD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
        fftwPlan = std::shared_ptr<void>(p, [](void* q) { fftwf_destroy_plan(static_cast<fftwf_plan>(q)); });
        impls.push_back({"fftw", [p, &buf] {
                             for (std::size_t b = 0; b < buf.frames; ++b) {
                                 fftwf_execute_dft(p, reinterpret_cast<fftwf_complex*>(buf.x + b * buf.N), reinterpret_cast<fftwf_complex*>(buf.y + b * buf.N));
                             }
                         }});
    } else {
        auto* p  = fftw_plan_dft_1d(int(N), reinterpret_cast<fftw_complex*>(buf.x), reinterpret_cast<fftw_complex*>(buf.y), FFTW_FORWARD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
        fftwPlan = std::shared_ptr<void>(p, [](void* q) { fftw_destroy_plan(static_cast<fftw_plan>(q)); });
        impls.push_back({"fftw", [p, &buf] {
                             for (std::size_t b = 0; b < buf.frames; ++b) {
                                 fftw_execute_dft(p, reinterpret_cast<fftw_complex*>(buf.x + b * buf.N), reinterpret_cast<fftw_complex*>(buf.y + b * buf.N));
                             }
                         }});
    }
#endif

    if (!opt.impls.empty()) {
        std::erase_if(impls, [&](const Impl& im) { return ("," + opt.impls + ",").find("," + im.name + ",") == std::string::npos; });
    }
    buf.fill(); // FFTW_MEASURE planning overwrites the arrays it is given
    for (auto& im : impls) { // accuracy, calibration, warm-up
        std::fill_n(buf.y, buf.N, std::complex<T>{});
        im.batch();
        const double err = relError(buf.x, buf.y, N);
        std::fprintf(acc, "%s,%s,%s,%zu,%.3e\n", opt.label.c_str(), im.name.c_str(), prec, N, err);
        if (opt.check && !(err < (sizeof(T) == 4 ? 2e-6 : 4e-15))) {
            std::fprintf(stderr, "FAIL: %s %s N=%zu relative error %.3e\n", im.name.c_str(), prec, N, err);
            opt.failed = true;
        }
        im.reps = 1;
        while (timeOnce(im) * 1e3 < opt.minSampleMs) {
            im.reps *= 2;
        }
        timeOnce(im); // discarded warm-up sample
    }
    std::vector<std::size_t> order(impls.size());
    for (int round = 0; round < opt.samples; ++round) {
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::shuffle(order.begin(), order.end(), shuffleRng);
        for (std::size_t i : order) {
            const double t = timeOnce(impls[i]) / double(impls[i].reps) / double(buf.frames);
            std::fprintf(csv, "%s,%s,%s,%zu,%d,%.3f\n", opt.label.c_str(), impls[i].name.c_str(), prec, N, round, t * 1e9);
        }
    }
    std::fflush(csv);
    std::fprintf(stderr, "  %-6s N=%-8zu done (%zu implementations)\n", prec, N, impls.size());
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto                   next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--samples") o.samples = std::stoi(next());
        else if (a == "--min-sample-ms") o.minSampleMs = std::stod(next());
        else if (a == "--min-log2") o.minLog2 = std::stoul(next());
        else if (a == "--max-log2") o.maxLog2 = std::stoul(next());
        else if (a == "--precision") { const auto p = next(); o.doFloat = p != "double"; o.doDouble = p != "float"; }
        else if (a == "--out") o.out = next();
        else if (a == "--label") o.label = next();
        else if (a == "--impls") o.impls = next();
        else if (a == "--check") o.check = true;
        else {
            std::fprintf(stderr, "usage: fftbench [--samples 40] [--min-sample-ms 5] [--min-log2 6] [--max-log2 20] [--precision float|double|both] [--label L] [--impls fastfft,fastfft-cx,gr4-SimdFFT,fftw] [--check] [--out raw.csv]\n");
            std::exit(a == "--help" ? 0 : 1);
        }
    }
    return o;
}

} // namespace

int main(int argc, char** argv) {
    Options opt = parse(argc, argv);
    std::fprintf(stderr, "fftbench: simd backend %s, %zu floats / %zu doubles per vector\n", FASTFFT_SIMD_BACKEND, fastfft::detail::Ops<float>::W, fastfft::detail::Ops<double>::W);
    if (opt.label.empty()) {
        opt.label = "w" + std::to_string(fastfft::detail::Ops<float>::W * 4);
    }
    std::FILE* csv = std::fopen(opt.out.c_str(), "w");
    std::FILE* acc = std::fopen((opt.out.substr(0, opt.out.rfind('.')) + "_accuracy.csv").c_str(), "w");
    if (!csv || !acc) {
        std::perror("fftbench: cannot open output");
        return 1;
    }
    std::fprintf(csv, "label,impl,precision,n,round,ns\n");
    std::fprintf(acc, "label,impl,precision,n,rel_err\n");
    std::mt19937 rng(12345);
    for (std::size_t l = opt.minLog2; l <= opt.maxLog2; ++l) {
        const std::size_t N = std::size_t{1} << l;
        if (opt.doFloat) runSize<float>(opt, N, csv, acc, rng);
        if (opt.doDouble) runSize<double>(opt, N, csv, acc, rng);
    }
    std::fclose(csv);
    std::fclose(acc);
    return opt.failed ? 2 : 0;
}
