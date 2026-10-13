# Design: a measured, portable FFT kernel family for GNU Radio 4

Status: draft for discussion, 2026-10-13.
Scope: the gr4 patch only; the paper and the standalone experiment stay on `main`.
Every claim is labelled **measured** (with its source) or **inferred** (reasoning or literature, not checked on our hosts).

## 1 · Goals and constraints

The patch follows three principles, in this order of evidence:

- **performance:** faster than today's `SimdFFT` wherever it is enabled, never slower;
- **measured:** every routing decision ("which kernel for which size, type and width") comes from a benchmark that ships with gr4, not from our judgement;
- **portability:** no ISA intrinsics, one source for every target, correct everywhere, fast where the `std::simd` backend is real.

Constraints from gr4 (`CLAUDE.md`, `CORE_DEVELOPMENT_GUIDELINE.md`, gr4-present `CLAUDE.md`):

- C++23, GCC 15+ with libstdc++, Clang 20+ with libc++, Emscripten; `vir::simd`/`std::simd`, no intrinsics; no exceptions in library code; header-only for hot paths;
- every public type has a `qa_` file with an independent ground truth, every hot path a `bm_` file on gr4's `benchmark.hpp`;
- no abstraction before its second real use, no settings "for later", names carry meaning, comments only for the why;
- commits are signed off by the human author and carry no assistant trailer;
- a C++26 preview path on GCC trunk (`<simd>`, `-std=c++26`) that uses only standard `std::simd`.

Facts about the targets that shape the design:

- gr4's library targets set no `-march`; only `bench/` and `algorithm/benchmarks/` add `-march=native` (`bench/CMakeLists.txt:13`, `algorithm/benchmarks/CMakeLists.txt`).
  A distribution build therefore gets SSE2: `native_simd<float>` has 4 lanes and `native_simd<double>` 2.
  The benchmarks measure a configuration most users never run. *Measured* by reading the build files.
- On Clang with libc++ and on Emscripten, vir-simd uses its array fallback, where our kernel is several times slower than `SimdFFT` (paper, libc++ table). *Measured.*
- `FFT::transformBluestein` computes three power-of-two FFTs through the same `FFT` object (`fft.hpp:353–380`), so any faster power-of-two route also speeds up non-power-of-two sizes. *Inferred* from the code.

## 2 · State of the art, and which machine parameter each idea optimises

An FFT on a current CPU is rarely limited by arithmetic.
It is limited by one of: the number of passes over memory, the cost of moving data between SIMD lanes, register pressure, or cache and TLB capacity.
Each family below attacks one of these.
References are in §10; they are cited from memory and not re-read for this note.

| family | idea | what it minimises | machine parameter it targets | where it wins |
| --- | --- | --- | --- | --- |
| radix-2 Cooley–Tukey, in place [1] | log2 N butterfly passes plus a bit-reversal permutation | code size | none | teaching, fallbacks (gr4's scalar path) |
| higher radix (4, 8, 16) [1, 3] | fuse 2–4 radix-2 levels into one pass | passes over memory, loads and stores | register count R: radix 16 needs about 16 + temporaries vector registers | everything that fits in registers; radix 16 wants R = 32 (AVX-512, NEON, SVE) |
| split radix [4], modified split radix [5] | mix radix 2 and 4 per level | real flops (≈ 4 N log2 N − 6 N vs 5 N log2 N) | FP throughput | historically; on SIMD the irregular structure costs more than the saved flops |
| Stockham autosort [2, 3] | out of place, every pass writes in natural order | the bit-reversal pass, strided access | SIMD width W, prefetchers: every access is unit stride | in-cache sizes on wide SIMD; GPUs (gr4's `SyclFFT` uses it) |
| Pease / constant geometry [15] | the same data flow in every pass | control and address logic | hardware pipelines, GPUs | rarely on CPUs |
| four-step / six-step [6, 19] | N = N1·N2: column FFTs, twiddle, row FFTs (six-step adds explicit transposes) | passes over main memory (2–3, independent of N) | last-level cache size, memory bandwidth, TLB reach | N beyond L2 |
| recursive, cache-oblivious [7, 8] | depth-first recursion so that every sub-FFT eventually fits a cache level | misses at every level without knowing the sizes | the whole cache hierarchy | large N on unknown machines; FFTW's default shape |
| generated codelets [7, 9] | straight-line code for small DFTs, scheduled for registers | loop overhead, spills | R, instruction latency | small N (≤ 256), and as leaves of the families above |
| planners and autotuning [7, 9] | search over the decompositions above, by measurement | the gap between a model and the actual machine | all of them, empirically | FFTW (`FFTW_MEASURE`), SPIRAL; our switch table is a static, offline version of this |
| vectorisation across butterflies [17] | lane j holds butterfly j of the same pass | scalar code | W | strides ≥ W; the first passes need an in-register transpose (our `transpose8`) |
| split complex layout (re[] and im[] separate) [13] | complex multiply without lane swaps | shuffles per complex product | shuffle throughput, W | narrow SIMD with cheap FMA and dear permutes; pffft/`SimdFFT` internally |
| interleaved complex layout | (re, im) pairs, multiply via swap and sign | conversion passes at the interfaces | memory passes | when the caller hands over `std::complex<T>`, as gr4 does |
| vectorisation across transforms ("batch", "vertical") [14] | lane j holds transform j; F frames at once | all lane shuffles inside the FFT | W, number of independent frames F ≥ W | many small FFTs per call: exactly gr4's streaming `FFT` block |
| real-input transforms [16] | N real points as an N/2 complex FFT plus one split pass | half the work for real signals | — | real input, gr4's spectrum blocks |
| mixed radix 3, 5, 7; prime factor [12]; Rader [11]; Bluestein [10] | sizes that are not powers of two | support of arbitrary N | — | user-chosen N such as 1000; gr4 has radix 2/3/5 in `SimdFFT` and Bluestein |
| reduced precision (FP16, BF16) | twice the lanes | bandwidth, FP throughput | AVX-512 FP16, Arm FP16 | displays and detection at about 1e-3 accuracy |
| multithreading | parallel four-step, or frames across cores | wall time | cores, memory bandwidth | large N or high frame rates; in gr4 the scheduler owns the threads |

Reference implementations and their licences, for reading and as yardsticks:
FFTW 3 [7] (GPL, yardstick only), Intel MKL and IPP (proprietary), pocketfft [14] (BSD, vectorises across transforms), pffft [13] (BSD, the origin of `SimdFFT`, 4 lanes, split layout inside), muFFT and OTFFT (MIT and BSD-like, Stockham with intrinsics), VkFFT (MIT, GPUs).
Only BSD/MIT code may inspire code in the patch; GPL code may not even be paraphrased.

## 3 · Where our kernel stands, and the gaps that motivate new variants

Our kernel (`gr4-pr` branch, `kernel-design.md`) is: Stockham autosort, interleaved complex data, radix 8 first pass with an in-register transpose, radix 16 middle passes vectorised over the stride, a radix 2/4/8/16 final pass, native width, complex float and double, powers of two N ≥ 64.

What the measurements say (paper and `results/`, two hosts, 120 samples per cell, 95 % bootstrap intervals):

| gap | evidence | likely cause | variant that targets it |
| --- | --- | --- | --- |
| G1 small N | FFTW is 9–21 % faster at N ≤ 512 on the HX 370 (float, 512 bit). *Measured.* | per-pass overhead and the first-pass transpose dominate when a pass is a few dozen vectors. *Inferred.* | V3 batch, V4 small-N codelets |
| G2 narrow SIMD, large N | at 128 bit our kernel is 0.54–0.71× of `SimdFFT` above 2^14 on the HX 370. *Measured.* | C = 2 complex values per vector: the radix-16 pass spills on 16 registers and every product pays a swap; `SimdFFT` uses split layout and radix 4. *Inferred.* | V2 split layout, V6 radix plan per width |
| G3 L2 boundary | FFTW wins at 2^11–2^12 and 2^18 on the VM, `vec<complex>` wins from 2^16 (halved twiddle tables). *Measured.* | twiddle-table and buffer traffic once in + out + twiddles leave L1 or L2. *Inferred.* | V5 four-step, V6 radix plan, compact twiddles |
| G4 double, mid N | FFTW is 15–40 % faster at 1k–256k. *Measured.* | not tuned for double at all; C halves, so the radix choice made for float is probably wrong. *Inferred.* | V6 radix plan per type |
| G5 emulated `simd` | Clang + libc++ and Emscripten: kernel disabled, `SimdFFT` stays. *Measured.* | vir's array fallback turns shuffles into element loops. *Measured* from generated code. | B2 backend (below), or nothing: the switch table routes to `SimdFFT` |
| G6 real input | the patch covers complex input only; the spectrum blocks often get real input. | — | V7 R2C |

The distribution-build case (W = 4 float, W = 2 double) is where G2 and G4 meet, and it is the configuration most gr4 users run.
Any variant must therefore be measured at 128 bit, not only at `-march=native` on an AVX-512 host.

## 4 · Candidate variants as a function of the input parameters

### Parameters the dispatch can see

| symbol | meaning | known at |
| --- | --- | --- |
| T | `float` or `double`; C2C or R2C | compile time |
| W | lanes of `native_simd<T>` (2 … 16) | compile time (`-march`) |
| R | vector registers: 16 (SSE, AVX2, WASM) or 32 (AVX-512, NEON, SVE) | compile time, derived from the target |
| B | `simd` backend: native (libstdc++, C++26 `<simd>`) or emulated (vir fallback) | compile time |
| N | transform size; classes below | run time (`resize`) |
| F | frames per call (`processBulk` chunk / N) | run time, per call |

Size classes, for complex float (halve the bounds for double): *tiny* N ≤ 256; *L1* N ≤ 2^11 (in + out ≈ 16 B per point against 32–48 KiB of L1d); *L2* N ≤ 2^16; *large* N > 2^16.
The bounds are *inferred* from typical cache sizes; the benchmark finds the real crossovers.

### The candidates

| id | variant | change against V1 | expected effect | cost |
| --- | --- | --- | --- | --- |
| V0 | `SimdFFT` (pffft port) | baseline, stays as the fallback | radix 2/3/4/5, real and complex, 4 lanes | none |
| V1 | Stockham, interleaved, radix 8 → 16 → tail | the existing kernel | measured above | none |
| V2 | Stockham, split layout inside | de-interleave folded into the first pass, re-interleave into the last | removes the swap per product; frees registers at C = 2 | first and last pass get heavier; second set of passes |
| V3 | batch across frames | lane j = frame j for F ≥ W frames; transpose W frames in the first pass and back in the last | no shuffles in the inner passes at all | needs F ≥ W; adds latency of W frames; only for the streaming block |
| V4 | small-N codelets | one or two passes, fully unrolled, for N ∈ {64 … 512} | removes per-pass loop and plan overhead | code size; one template per size |
| V5 | four-step for large N | N = N1 · N2 with blocked transposes, V1 as the sub-FFT | 2–3 passes over memory instead of log16 N | an extra transpose; only above L2 |
| V6 | radix plan per (T, W, N) | choose the pass radices (e.g. 8·16·16 vs 16·16·8 vs 4·…) from a table | fixes G2, G3, G4 where the cause is the plan, not the kernel | none in code: a table; the benchmark measures the plans |
| V7 | real input (R2C) | N/2 complex FFT with V1 plus a split pass | ≈ 2× for real signals | one extra pass and its twiddles |
| B2 | GNU vector extension backend | `__attribute__((vector_size))` types where `simd` is emulated | real SIMD on Clang/libc++ and Emscripten without intrinsics | a second backend of five primitives; arguably not "std::simd only" |

Deferred, with the reason: mixed radix 3/5 (V0 already covers them, and Bluestein inherits the speed of the power-of-two route); FP16 (no `std::simd<std::float16_t>` on our compilers yet, *inferred*); multithreading (the gr4 scheduler owns threads; better as a graph-level split).

### Which candidate to measure in which regime

Expected winner first; every cell lists what the benchmark must compare.

| regime | tiny | L1 | L2 | large |
| --- | --- | --- | --- | --- |
| W ≥ 8, R = 32 (AVX-512) | V4, V3, V1 | V1, V3, V6 | V1, V6 | V5, V1 |
| W = 8, R = 16 (AVX2) | V4, V3, V1 | V1, V2, V6 | V1, V2, V6 | V5, V2 |
| W = 4, R = 16 (SSE2 distro build, WASM) | V3, V0, V4 | V2, V0, V1 | V0, V2, V6 | V0, V5 |
| W = 4, R = 32 (NEON) | V3, V4, V1 | V1, V2 | V1, V2, V6 | V5, V0 |
| emulated backend B | V0 | V0 | V0 | V0 (or B2) |

All rows apply per T; double halves W and moves every column boundary down by one octave.

### What to try first, and why

1. **V6 radix plans**: no new kernel code, and it may close G2–G4 on its own; it also forces the benchmark to become a real tuning tool.
2. **V2 split layout**: the only candidate aimed at the distribution build, which is the majority case and today's worst loss (0.54× at 128 bit).
3. **V3 batch across frames**: the streaming `FFT` block always has many frames per call, so this is the one place where gr4's structure gives an advantage FFTW-style libraries cannot use; it targets G1.
4. **V5 four-step**: an earlier AVX-512 prototype of ours had it (branch `fastfft-avx512`), so its cost is known; it targets the large-N and double gaps.
5. **V7 R2C** next, because the spectrum blocks need it, but it is a feature, not a tuning variant.

V4 and B2 wait until the benchmark shows that V3 or the routing to V0 leave a gap worth their code.

## 5 · Routing: the implementation switch table

`gr::algorithm::FFT::compute` already routes between `SimdFFT` and the scalar path.
The patch extends that one function and adds no dispatcher type.

```cpp
// measured: bm_FFT on <host>, <date>, <results file>; regenerate with `bm_FFT --routes`
constexpr std::array kRoutesFloat16 = {Route{.maxLog2N = 9, .kernel = Kernel::stockham, .plan = {8, 8}}, ...};
```

- One table per (T, W) that has been measured; an unmeasured (T, W) uses the nearest measured W of the same T with R taken into account, else `SimdFFT`.
  The tables are `constexpr`, so the choice costs a comparison per `resize`, not per call.
- An entry changes only when the benchmark shows a difference whose 95 % interval excludes 1.
  Otherwise the existing route stays, so noise cannot churn the table.
- Each table carries a one-line comment naming its measurement, as gr4 requires for non-obvious choices.
- `useSimdFFT` remains the switch that restores the old behaviour.

## 6 · What gr4's benchmark should learn from ours

Today gr4 has `algorithm/benchmarks/bm_fft.cpp` (the `FFT` dispatcher, `benchmark<1000>`), `bm_FFT_backends.cpp` (`SimdFFT` against SYCL devices, `benchmark<10>`) and `bm_pffft.cpp` (the pffft upstream benchmark).
All three run on `bench/benchmark.hpp`, which times each iteration with `high_resolution_clock`, reports min, mean, stddev, median and max, prints perf-counter instructions per operation, and renders a table.

| aspect | gr4 today | our harness | proposal for gr4 |
| --- | --- | --- | --- |
| sample length | one call per sample: ≈ 150 ns at N = 512, close to the clock's resolution and overhead | calibrated so that one sample lasts ≥ 5 ms | `benchmark.hpp`: optional minimum sample duration; the iteration count per sample is calibrated once |
| warm-up | one untimed `compute` | one discarded sample after calibration | discard the first sample |
| what is timed | `output = fft.compute(signal)`, which allocates a fresh output vector per call (`fft.hpp:155–159`) | the transform into preallocated buffers | time `compute(in, out)` into a preallocated output |
| cache state | the same hot frame every iteration | consecutive frames of a ≥ 2^16-point buffer, as the streaming block sees them | both modes, named: `hot` and `stream` |
| order and drift | implementations run one after another | fresh random order in every round, 40 rounds | a comparison group: variants of one size run interleaved in shuffled order |
| statistic | mean and stddev (inflated by one-sided noise) | median with a 95 % bootstrap interval, relative MAD | median with a distribution-free 95 % interval from order statistics (binomial), relative MAD; no resampling, a few lines |
| comparison | none | paired same-round ratios with an interval | the ratio of each variant to the current route, with its interval |
| scaling | `N·ln N` in `bm_fft`, `5·N·log2 N` in `bm_FFT_backends` | ns per transform | one convention: `5·N·log2 N` (FFTW's), plus ns per transform |
| accuracy | one bin within 0.1 of the expected magnitude | relative L2 error against a long-double DFT, before timing | a `boost::ut` expectation against a long-double DFT per size; a failing kernel is not timed |
| sizes | 512, 1024, 8192, 65536, 1009 | 2^6 … 2^20 | 2^6 … 2^20 plus 1009 and 1000; a short default list for CI, the full sweep on request |
| host record | none | `host.txt`: CPU, governor, load, SMT, compiler, flags, pinning | one header line: compiler, flags, `simd` backend, W per type, CPU model |
| instructions | perf counters, instructions per operation | none | keep, and print them per transform: gr4's agent rules gate hot paths on instruction count |
| processes, pinning | none | 3 processes, `taskset`, run script | documented invocation (`taskset -c <cpu> ./bm_FFT --rounds 40`), no script in gr4 |
| output | a terminal table | raw CSV plus a Python analysis | the table as today, plus `--routes`: the switch table this host would choose, ready to paste |
| yardstick | none | FFTW 3.3.10 | stays outside gr4 (GPL); our repository keeps the FFTW comparison |

The generic parts (minimum sample duration, warm-up discard, median interval, MAD, comparison group) belong in `benchmark.hpp`, as an opt-in that leaves every existing `bm_` unchanged.
They go in a commit of their own, because they help every benchmark, not only the FFT.
The FFT-specific parts (variants, accuracy pre-check, `stream` mode, `--routes`) replace `bm_fft.cpp`; `bm_FFT_backends.cpp` stays as is, apart from routing its CPU row through `FFT` rather than `SimdFFT` directly.

## 7 · C++23 and C++26 paths

One kernel source; the backends differ only in five primitives: load, store, two-input zip, re/im swap, complex broadcast.

| path | toolchain | `simd` type | two-input shuffle | status |
| --- | --- | --- | --- | --- |
| C++23, native | GCC 15+ (libstdc++), via vir | `stdx::native_simd<T>` | `__builtin_shufflevector` on the same-width vector type (a bit cast, free) | measured: level with the hand-written AVX-512 kernel |
| C++23, emulated | Clang 20+ with libc++, Emscripten | vir array fallback | element loops | routed to `SimdFFT` by the table; B2 is the alternative |
| C++26 preview [18] | GCC trunk (17), `-std=c++26`, `__cpp_lib_simd` | `std::simd::vec<T>` | `std::simd::permute(std::simd::cat(a, b), idx)` | measured: level on the HX 370, 5–25 % slower at N ≤ 512 on the VM |

The C++26 path is selected by `__cpp_lib_simd` alone and has no compiler extensions.
`vec<std::complex<T>>` stays out of the patch: it is slower up to 2^15 and needs `bit_cast` workarounds (paper); it is worth revisiting as GCC's implementation matures.

## 8 · Naming

`SimdFFT` describes every SIMD FFT, and `FastFFT` every FFT anyone would ship; neither tells a reader what is inside or when to prefer it.
gr4's rule is *nomen est omen*: the name should carry the algorithm.

| option | for | against |
| --- | --- | --- |
| `StockhamFFT` (chosen 2026-10-13) | names the algorithm all our variants share (V2–V6 are layouts, plans and decompositions of Stockham passes); matches the "Van Loan Stockham" wording in gr4's `SyclFFT` | V5 four-step is a Stockham sub-FFT inside another scheme |
| `NativeWidthFFT` | names the actual difference from `SimdFFT` | stops being true the day `SimdFFT` is widened |
| `fft::Stockham`, `fft::Pffft` in a namespace | symmetric, and room for both kernels by lineage | renames shipped API (`SimdFFT`); a larger discussion with the maintainers |
| keep `FastFFT` | continuity with the paper | names a property every FFT claims |

Renaming `SimdFFT` itself (to `PffftFFT` or similar) would make the pair symmetric, but it changes shipped API and belongs in a separate proposal to the maintainers.

## 9 · Branch layout and next steps

`gr4/patches/` is the series as `git format-patch` output against the gnuradio4 revision pinned in `gr4/REVISION`; `gr4/apply.sh` clones and applies it.
The gr4-bound commits are prepared for Dennis to sign off (`git commit -s`), one per change scenario, with no assistant trailer.

1. Agree on this design (name decided: `StockhamFFT`).
2. `benchmark.hpp` opt-in statistics and comparison groups, with a `qa_` for the median interval.
3. Port V1 under the agreed name, C++23 and C++26 paths, `qa_` against a long-double DFT; the new `bm_fft.cpp` with `--routes`.
4. Measure V6 plans at W = 4, 8, 16 on both hosts; first switch tables.
5. V2, then V3, then V5, each kept only if the benchmark shows a gain outside the noise.

## 10 · Measured outcomes (Xeon @ 2.1 GHz AVX-512 VM, GCC 15.3, 2026-10-13)

Run 2 of `bm_fft` (stream mode, 40 interleaved rounds, 5 ms samples) at `-march=x86-64`, `x86-64-v3` and `native`, plus the C++26 path on GCC trunk; raw output in `results/xeon-20261013/run2/`.
Time of the best StockhamFFT plan relative to `SimdFFT`, range over N = 2^6 .. 2^20:

| target | float | double |
| --- | ---: | ---: |
| 128 bit (SSE2) | 0.31 – 0.81 | 0.57 – 0.91 |
| 256 bit (AVX2) | 0.44 – 0.77 | 0.62 – 1.00 |
| 512 bit (AVX-512) | 0.32 – 0.65 | 0.54 – 0.80 |
| 512 bit, C++26 `std::simd` | 0.34 – 0.66 | 0.56 – 0.78 |

What the measurements decided:

- **V1 + V6** win everywhere measured; the route tables in `FFTRoutes.hpp` are this run's `bm_fft` output, so `FFT` never routes to `SimdFFT` on this host.
  The earlier 128-bit rule (SimdFFT above 2^14, HX 370) predates the two fixes below and has to be re-measured there.
- **V2 split** beats the best interleaved plan only at 128 bit for N ≥ 2^15 (float 0.52–0.65, double 0.68–0.81), and loses at small N and at 512 bit; it stays next to the benchmark, not routed.
- **V3 batch** loses almost everywhere (up to 3.6× slower than V1): the in-register transpose of W frames costs more than the shuffles it saves. Dropped.
- **V5 four-step** never beats the best V1 plan up to 2^20. Dropped; it may return above 2^20.
- Two causes of slow double code, both fixed in the StockhamFFT commit: gr4's Release `-O2` keeps fixed-count loops rolled (`#pragma GCC unroll`), and at two lanes the 4-scalar bridge vector is not trivially copyable, so swaps went through the stack.
- The accuracy gate found an upstream bug: every non-power-of-two `FFT` (Bluestein) was wrong; fixed with tests as the first commit.

## 11 · References

1. J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series," *Math. Comput.*, vol. 19, pp. 297–301, 1965.
2. T. G. Stockham, "High-speed convolution and correlation," in *Proc. AFIPS Spring Joint Comput. Conf.*, 1966, pp. 229–233.
3. C. Van Loan, *Computational Frameworks for the Fast Fourier Transform*. Philadelphia, PA, USA: SIAM, 1992.
4. P. Duhamel and H. Hollmann, "Split radix FFT algorithm," *Electron. Lett.*, vol. 20, no. 1, pp. 14–16, 1984.
5. S. G. Johnson and M. Frigo, "A modified split-radix FFT with fewer arithmetic operations," *IEEE Trans. Signal Process.*, vol. 55, no. 1, pp. 111–119, 2007.
6. D. H. Bailey, "FFTs in external or hierarchical memory," *J. Supercomput.*, vol. 4, no. 1, pp. 23–35, 1990.
7. M. Frigo and S. G. Johnson, "The design and implementation of FFTW3," *Proc. IEEE*, vol. 93, no. 2, pp. 216–231, 2005.
8. M. Frigo, C. E. Leiserson, H. Prokop and S. Ramachandran, "Cache-oblivious algorithms," in *Proc. 40th IEEE FOCS*, 1999, pp. 285–297.
9. M. Püschel *et al.*, "SPIRAL: Code generation for DSP transforms," *Proc. IEEE*, vol. 93, no. 2, pp. 232–275, 2005.
10. L. I. Bluestein, "A linear filtering approach to the computation of discrete Fourier transform," *IEEE Trans. Audio Electroacoust.*, vol. 18, no. 4, pp. 451–455, 1970.
11. C. M. Rader, "Discrete Fourier transforms when the number of data samples is prime," *Proc. IEEE*, vol. 56, no. 6, pp. 1107–1108, 1968.
12. I. J. Good, "The interaction algorithm and practical Fourier analysis," *J. R. Stat. Soc. B*, vol. 20, no. 2, pp. 361–372, 1958.
13. J. Pommier, "PFFFT: a pretty fast FFT," 2013. [Online]. Available: https://bitbucket.org/jpommier/pffft
14. M. Reinecke, "pocketfft," 2019. [Online]. Available: https://github.com/mreineck/pocketfft
15. M. C. Pease, "An adaptation of the fast Fourier transform for parallel processing," *J. ACM*, vol. 15, no. 2, pp. 252–264, 1968.
16. H. V. Sorensen, D. L. Jones, M. T. Heideman and C. S. Burrus, "Real-valued fast Fourier transform algorithms," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 35, no. 6, pp. 849–863, 1987.
17. F. Franchetti and M. Püschel, "Short vector code generation for the discrete Fourier transform," in *Proc. IEEE IPDPS*, 2003.
18. M. Kretz, "std::simd — merge data-parallel types from the Parallelism TS 2," ISO/IEC JTC1/SC22/WG21 P1928, 2024.
19. W. M. Gentleman and G. Sande, "Fast Fourier transforms — for fun and profit," in *Proc. AFIPS Fall Joint Comput. Conf.*, 1966, pp. 563–578.
