# Faster FFT for the gnuradio4 FFT block

A hand-written AVX-512 complex<float> FFT (`fastfft/FastFFT.hpp`, header-only, no dependencies) and a drop-in
`FastFFT<float>` block (`fastfft/FastFFTBlock.hpp`) with the same ports, settings and semantics as the streaming
`gr::blocks::fft::FFT<float>` block of [gnuradio4](https://github.com/fair-acc/gnuradio4) (rev `af62235`), which uses
the library's `SimdFFT` (a pffft derivative, 4-wide vectors).

`gr4/bm_fft_fast.cpp` benchmarks both blocks with gnuradio4's own benchmark facility (`benchmark.hpp`, as in
`bm_fft_block`). It feeds identical random input through `processBulk`, checks the two outputs agree and both match a
double-precision reference, then times them. Each timed call processes 65536 samples (or one frame if N is larger).

## Results

Host: Intel Xeon (Sapphire Rapids class, AVX-512 + FP16), 4 vCPU VM, pinned to one core; clang 20.1, libc++,
`-O3 -march=native`. Throughput is FFTW's convention, 5·N·log2(N) flop per transform. Full output in `results/`.

| N | block SimdFFT | block FastFFT | speedup | kernel SimdFFT | kernel fastfft | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 7.5 GF/s | 30.6 GF/s | 4.1x | 7.9 | 42.8 | 5.4x |
| 256 | 10.5 | 40.4 | 3.8x | 11.6 | 56.7 | 4.9x |
| 1024 | 11.2 | 47.1 | 4.2x | 12.6 | 68.3 | 5.4x |
| 4096 | 10.7 | 40.5 | 3.8x | 11.4 | 48.6 | 4.3x |
| 16384 | 10.2 | 42.6 | 4.2x | 11.6 | 55.0 | 4.7x |
| 65536 | 10.7 | 30.3 | 2.8x | 11.4 | 37.4 | 3.3x |
| 262144 | 8.5 | 19.6 | 2.3x | 8.7 | 24.4 | 2.8x |
| 1048576 | 8.8 | 18.0 | 2.0x | 9.6 | 21.6 | 2.3x |

"block" includes the gr4 port copy-in/read-back; "kernel" is the bare transform on the same aligned buffers.

Accuracy (max |error| / max |X|, vs. double reference): SimdFFT 1.0e-7 … 3.8e-7, fastfft 1.0e-7 … 2.2e-7. The two blocks
agree to ≤ 4.2e-7 at every size.

As a yardstick, FFTW 3.3.10 built from source with AVX-512 codelets, `FFTW_MEASURE`
(`results/standalone_vs_fftw.txt`): fastfft is level with it at 256–4096 and 64k–256k (±7%), 10–35% faster at 64–128,
512 and 8k–32k, and 1.7–1.9x faster at 512k–1M.

## How it works (iterations)

1. Radix-4 Stockham autosort, interleaved complex in zmm registers, `fmaddsub`-based complex multiply, first two
   stages vectorised over the butterfly index with in-register transposes. Already 3–4x over SimdFFT up to 16k.
2. Four-step (N = N1·N2) with "lane-vectorised" sub-FFTs: 8 independent columns per vector, so no shuffles and only
   broadcast twiddles, plus an 8x8 register transpose. Two memory passes for any N; wins at N ≥ 1M.
3. Radix-8 Stockham: first stage writes through an 8x8 transpose, later stages vectorise over the stride.
4. Radix-16 stages (4x4 in registers) after the first radix-8 stage: fewer passes, fewer twiddle multiplies; 10–20%
   on top of 3. Padding the four-step scratch matrix removed power-of-two cache-set aliasing (−20% at 256k).

Dispatch: N < 64 radix-4, 64 ≤ N < 2^20 radix-8/16 Stockham, N ≥ 2^20 four-step. Gains plateaued at parity with
FFTW's AVX-512 codelets.

## Limits

- complex<float>, power-of-two sizes only (the block's `fft_size` already must be a power of two); `FFT<double>` and
  the `DataSet` spectrum mode still use gr4's code.
- Needs AVX-512F/DQ (compile with `-march=native` on such a host). No fallback for other ISAs.
- Inverse uses the block's existing conj → forward → conj/N path, unchanged.

## Build

```sh
scripts/build_gr4.sh            # clones gnuradio4 at the pinned rev, adds bm_fft_fast, builds it
taskset -c 2 gnuradio4/build/blocks/fourier/benchmarks/bm_fft_fast
FFTW_PREFIX=/path/to/fftw-avx512 scripts/build_standalone.sh   # accuracy test + timing vs FFTW
```

`bin/bm_fft_fast` is the binary the results came from (x86-64 with AVX-512, needs libc++ 20 at runtime).

## native_simd port (`fastfft/FastFFTSimd.hpp`)

The same algorithm written against `native_simd<float>` (vir-simd, as gr4 uses), with no ISA intrinsics; the width
follows the target (16 floats on AVX-512, 8 on AVX2, 4 on SSE). Two helpers (re/im swap and a two-input zip) use the
compiler's generic vector extension (`__builtin_shufflevector`), because std::simd has no two-input shuffle and the
`concat`+`simd_permute` route lowers to element moves that cost ~3x the whole stage. `stdx::fma` is spelled `a*b+c`
(compile with `-ffp-contract=fast`) because libstdc++ does not inline it.

ns per transform, one core (`standalone/test_simd.cpp`):

| N | gcc 14 + vir (libstdc++ simd) | clang 20 + libstdc++ simd | clang 20 + vir fallback | hand-written AVX-512 |
|---:|---:|---:|---:|---:|
| 64 | 43 | 33 | 153 | 39–48 |
| 1024 | 598 | 654 | 5908 | 676–737 |
| 4096 | 4741 | 4926 | 34976 | 4567–4673 |
| 65536 | 132294 | 133755 | 798477 | 128344–133751 |
| 1M | 4.63M | 4.60M | 18.9M | 4.6–4.7M |

With a real simd implementation the portable version matches the hand-written kernel. On clang, vir-simd skips
`<experimental/simd>` and uses its own array-based fallback, which is 4–9x slower here. gr4's own SimdFFT shows the
same effect (`standalone/simdfft_compiler_cmp.cpp`): 2x faster built with gcc 14 than with clang 20 + libc++
(N=1024: 1980 vs 3832 ns; 4096: 11927 vs 20627 ns).

Build: `g++-14 -std=c++23 -O3 -march=native -ffp-contract=fast -Ifastfft -I<vir-simd> standalone/test_simd.cpp`, or with
clang: `-stdlib=libstdc++ -DFASTFFT_USE_STD_EXPERIMENTAL_SIMD` (no vir needed).
