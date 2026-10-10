#ifndef FASTFFT_BLOCK_HPP
#define FASTFFT_BLOCK_HPP

#include <gnuradio-4.0/fourier/fft.hpp>

#include "FastFFT.hpp"

namespace gr::blocks::fft {

/// drop-in for the CPU path of `FFT<float>` (raw streaming complex->complex), same ports, settings and semantics,
/// with the transform done by the hand-written AVX-512 `fastfft::Plan` instead of `SimdFFT`
template<typename T>
requires std::same_as<T, float>
struct FastFFT : gr::Block<FastFFT<T>, gr::Resampling<1UZ, 1UZ, false>, gr::Stride<>> {
    using ComplexType = std::complex<T>;

    PortIn<ComplexType>  in;
    PortOut<ComplexType> out;

    Annotated<gr::Size_t, "fft size", Limits<8UZ, 1048576UZ, &detail::isPowerOfTwo>> fft_size = 4096UZ;
    Annotated<bool, "inverse">                                                       inverse  = false;

    GR_MAKE_REFLECTABLE(FastFFT, in, out, fft_size, inverse);

    fastfft::Plan _plan;

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        refuseChunkSizeOverride(*this, newSettings, static_cast<gr::Size_t>(fft_size));
        this->input_chunk_size  = fft_size;
        this->output_chunk_size = fft_size;
    }

    gr::work::Status processBulk(InputSpanLike auto& inSpan, OutputSpanLike auto& outSpan) {
        const auto N      = static_cast<std::size_t>(fft_size);
        const auto frames = gr::windowGeometry(*this, inSpan.size(), outSpan.size());
        if (frames.nWindows == 0UZ) {
            std::ignore = inSpan.consume(0);
            outSpan.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        if (_plan.size() != N) {
            _plan.init(N);
        }

        const auto hop      = frames.hop;
        const auto nBatches = frames.nWindows;
        const auto total    = nBatches * N;

        for (std::size_t b = 0; b < nBatches; ++b) {
            const ComplexType* inSlice  = inSpan.data() + b * hop;
            ComplexType*       outSlice = outSpan.data() + b * N;
            if (inverse) {
                std::transform(inSlice, inSlice + N, outSlice, [](auto z) { return std::conj(z); });
                _plan.forward(outSlice, outSlice);
                T invN = T(1) / static_cast<T>(N);
                std::transform(outSlice, outSlice + N, outSlice, [invN](auto z) { return std::conj(z) * invN; });
            } else {
                _plan.forward(inSlice, outSlice);
            }
        }

        std::ignore = inSpan.consume(total);
        outSpan.publish(total);
        return work::Status::OK;
    }
};

} // namespace gr::blocks::fft

#endif
