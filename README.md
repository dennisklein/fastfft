# fastfft: a portable std::simd FFT for gnuradio4

> **About this repository.** This project started as an experiment: how far Claude Code can carry a focused
> performance-engineering task, from benchmarking an existing component to a measured and documented alternative, with
> a human steering and reviewing. The paper in `paper/` is part of the same experiment. It shows what kind of
> publication an AI produces when a human spends only a few hours of their own time on it. Please read both in that
> spirit. This is not meant as state-of-the-art FFT work, and the authors claim no particular expertise in FFT
> algorithms. The results are careful measurements on two machines, but they have not been checked exhaustively
> against the literature. The code, the measurements and most of the text came out of a single Claude Code session
> spanning about 15 hours, including measurement runs. It used one Claude model and no other AI tools, and took about
> 480 model requests, 0.4 million output tokens and 104 million input tokens, 99 % of them cached re-reads of the
> growing conversation. Dennis Klein posed the question, steered the work, reviewed the results and ran the
> measurements on the second host.

`include/fastfft/fft.hpp` is a single-header, power-of-two, complex forward FFT (float and double) written against
`std::simd` (C++26, or `std::experimental::simd` before that), without ISA intrinsics. On one AVX-512 core it is 1.6–3.4x faster than
[gnuradio4](https://github.com/fair-acc/gnuradio4)'s `SimdFFT` in single precision (1.3–2.0x in double), and
0.7–2.0x as fast as FFTW 3.3.10 with AVX-512 codelets, depending on size and precision. The paper in `paper/` explains
how and reports the measurements; this repository contains everything needed to repeat them on another Linux host.

```
include/fastfft/fft.hpp      the implementation (one header, ~450 lines)
include/fastfft/fft_cx.hpp   experimental variant on C++26 std::simd::vec<std::complex<T>> (GCC >= 17)
bench/fftbench.cpp           benchmark + accuracy check: fastfft, gnuradio4 SimdFFT, FFTW
bench/run.sh                 one-command reproduction on this host -> results/<host>-<date>/
bench/analyze.py             statistics: medians, bootstrap confidence intervals, paired speedups
bench/build_gcc_trunk.sh     builds GCC trunk (C++26 std::simd) into a local prefix, dnf or apt based distros
bench/gr4_compare.sh         builds gnuradio4 with and without the patch, runs its tests and block benchmarks
gr4/0001-fastfft.patch       patch for gnuradio4 (rev af62235) that uses fastfft in gr::algorithm::FFT
gr4/0002-bm_fft_stream.patch gnuradio4 benchmark for the streaming FFT block (used by gr4_compare.sh)
paper/                       LaTeX source, PDF, mkdata.py (results -> tables and figure data)
results/                     measurements, one directory per host and run
```

## Using the header

```cpp
#include <fastfft/fft.hpp>
fastfft::FFT<float> fft(4096);           // N a power of two >= 2; precomputes twiddles and scratch
fft.forward(in, out);                    // const std::complex<float>* -> std::complex<float>*, in == out allowed
```

Compile with C++20 or later, `-O3 -march=<target> -ffp-contract=fast`. The SIMD backend is C++26 `std::simd` when
available (see below), else libstdc++'s `<experimental/simd>` (GCC >= 12, or clang with `-stdlib=libstdc++`). With libc++ the header falls back to
[vir-simd](https://github.com/mattkretz/vir-simd), which on clang emulates the vector types with arrays and is several
times slower; `FASTFFT_NATIVE_SIMD` is 0 in that case. Sizes below 64 use a scalar radix-2 path. The transform is
unnormalised, `X[k] = sum_n x[n] exp(-2 pi i n k / N)`; for an inverse, conjugate input and output and scale by 1/N.

## Reproducing the measurements

```sh
bench/run.sh                       # defaults: FFTW built from source, 3 runs x 40 rounds, newest g++ found
bench/run.sh --fftw system         # use the distribution's libfftw3 + libfftw3f instead (often without AVX-512)
bench/run.sh --fftw none --runs 5  # skip FFTW, five process runs
bench/run.sh -- --max-log2 16      # arguments after -- go to fftbench
```

C++26 `std::simd` (GCC trunk, x86-64 only so far). No distribution packages a GCC with `<simd>` yet, so build one
first; it goes into `build-<host>/gcc-trunk`, takes ~15 min on 4 cores and ~3 GB, and installs build dependencies
with dnf or apt (`--no-deps` to skip):

```sh
bench/build_gcc_trunk.sh                                   # Fedora/RHEL/Alma/Rocky or Debian/Ubuntu
bench/run.sh --gcc-trunk build-$(hostname -s)/gcc-trunk    # adds labels gcc17, gcc17-exp and gcc17-cxbc
```

With `-std=c++26` and `<simd>` available, the header uses `std::simd::permute`/`cat` instead of the
`__builtin_shufflevector` helpers, i.e. no compiler extensions at all. `gcc17` is that build (all implementations,
plus `fastfft-cx` from `fft_cx.hpp`); `gcc17-exp` is the same GCC with `-DFASTFFT_NO_STD_SIMD` (the
`<experimental/simd>` path), which separates the API from the compiler; `gcc17-cxbc` builds `fastfft-cx` with
`-DFASTFFT_CX_EXPERIMENT_BITCAST`, a non-portable variant used only to attribute the cost of `fastfft-cx`.

Requirements: Linux, cmake >= 3.24, git, python3, GCC >= 14 (or clang with libstdc++ 14; gnuradio4's headers need
it), util-linux; for the FFTW build also make, a C compiler and curl. The script fetches gnuradio4 at the pinned
revision and vir-simd v0.4.4 (headers only, nothing of gnuradio4 is built), builds FFTW 3.3.10 with the SIMD codelets
the CPU supports (SSE2/AVX/AVX2/AVX-512 on x86-64, NEON on aarch64), compiles `fftbench` with `-O3 -march=native`, and
writes to `results/<hostname>-<date>/`:

| file | content |
|---|---|
| `host.txt` | date, kernel, compiler, flags, FFTW configuration, CPU governor/boost/SMT, load, `lscpu` |
| `check_accuracy.csv` | max error / max magnitude of every implementation vs. a long-double reference DFT |
| `raw-<r>.csv`, `width-raw-w<B>-<r>.csv`, `<label>-raw-<r>.csv` | every timing sample: label, impl, precision, n, round, ns |
| `summary.md`, `summary.csv`, `speedup.csv` | the statistics below |

A run without `--gcc-trunk` and `--libcxx` took about 10 minutes on the reference host. Please send back the whole results directory.

For cleaner numbers: run on an otherwise idle machine, fix the CPU frequency if you can (`cpupower frequency-set -g
performance`, disable turbo), and pick an isolated core with `--cpu N`. None of this is required, the method below
is designed to cope with ordinary noise and to show how much there was.

### Method

- **Workload.** Complex-to-complex forward transform, N = 2^6 … 2^20, float and double. Input is uniform random in
  [-1, 1)². One *sample* transforms max(N, 65536) consecutive values as frames of size N (a streaming workload, the
  way a gnuradio4 block sees it; up to 1 MiB of floats, so small sizes run from L2) and reports ns per transform.
  FFTW uses `FFTW_MEASURE` plans executed with `fftw_execute_dft` on each frame; SimdFFT and fastfft are called the
  same way through their public API.
- **Correctness first.** Every implementation's output is compared with a long-double radix-2 reference before it is
  timed; `--check` fails the run if the relative error exceeds 2e-6 (float) or 4e-15 (double).
- **Calibration and warm-up.** Each sample repeats the batch until it lasts at least 5 ms; one sample per
  implementation is discarded as warm-up.
- **Randomised, interleaved rounds.** 40 rounds per process; within a round every implementation is measured once,
  in a freshly shuffled order. Slow drifts (frequency, temperature, other tenants) therefore hit all implementations
  alike and samples from the same round can be paired. The whole process is repeated 3 times (`--runs`) so that
  between-process effects (memory placement, ASLR) enter the spread.
- **Statistics** (`analyze.py`, standard library only). Per cell the median over all 120 samples with a 95 %
  percentile-bootstrap confidence interval (4000 resamples) and the relative median absolute deviation. Speedups are
  the median of the per-round ratios t_baseline / t_fastfft with their own bootstrap CI. Medians, not means, because
  timing noise is one-sided (interruptions only add time).
- **Pinning.** All runs are pinned to one core with `taskset` (default: the last CPU).

## gnuradio4 patch

`gr4/0001-fastfft.patch` applies to gnuradio4 `af62235` (`git apply`). It adds the header as
`gnuradio-4.0/algorithm/fourier/fastfft.hpp` and

1. in `gr::algorithm::FFT::compute`, routes complex power-of-two transforms with N >= 64 to fastfft when the
   simd backend is native (`useFastFFT`, default `FASTFFT_NATIVE_SIMD`); everything else still goes to SimdFFT or the
   radix-2/Bluestein fallback;
2. in the spectrum block `FFT<T, DataSet<P>>`, computes into the preallocated output buffer instead of a new vector
   per frame, and fuses magnitude, phase and the fftshift into one pass (same functions, same results, no
   temporaries; the unwrap path keeps its original order).

`bench/gr4_compare.sh` reproduces the before/after comparison with gnuradio4's own benchmark facility (`bm_fft_stream`
from 0002, and the existing `bm_fft_block`) and runs the FFT unit tests on the patched tree. Use GCC: gnuradio4's
clang build uses libc++, where vir-simd only emulates simd, and the patch then keeps SimdFFT.

## Results so far

| host | kernel data | gr4 patch data |
|---|---|---|
| Intel Xeon VM, 4 vCPU, AVX-512, GCC 14.2 + GCC trunk (C++26) | `results/vm-20261010-2129/summary.md` | `results/vm-20261010-2000-gr4/summary.md` |
| AMD Ryzen AI 9 HX 370 (Zen 5), AVX-512, GCC 16.2 + GCC trunk (C++26) | `results/zeta-20261011-0058/summary.md` | |

## Building the paper

```sh
python3 paper/mkdata.py results/<kernel-dir> results/<gr4-dir> --host2 results/<second-host-dir>   # writes paper/numbers.tex, tables.tex, data/
cd paper && pdflatex fastfft.tex && pdflatex fastfft.tex
```

## Licence

Copyright 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH.

Authors: Dennis Klein <d.klein@gsi.de> and Claude (Anthropic).

The code (headers, benchmark, scripts, gnuradio4 patches) is licensed under the GNU Lesser General Public License v3.0
or later with the same static-linking exception as gnuradio4 (`LGPL-3.0-or-later WITH LGPL-3.0-linking-exception`),
so the patch can go upstream unchanged. The paper and the measurement data in `results/` are licensed under
CC BY 4.0. Every file carries an SPDX header or is covered by `REUSE.toml`; the licence texts are in `LICENSES/`
(`LICENSE` is the LGPL text; it supplements the GPL v3, whose text is in `COPYING`).
