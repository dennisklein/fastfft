#!/usr/bin/env bash
# Standalone kernel accuracy test and timing vs. FFTW. FFTW_PREFIX should point at an FFTW 3.3.10 build configured with
# --enable-float --enable-avx512 --enable-avx2 --enable-fma (the distro package has no AVX-512 kernels).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CXX=${CXX:-clang++-20}
FLAGS="-std=c++23 -O3 -march=native -mtune=native -I$HERE/fastfft"
$CXX $FLAGS "$HERE/standalone/test_accuracy.cpp" -o "$HERE/standalone/test_accuracy"
if [ -n "${FFTW_PREFIX:-}" ]; then FFTW="-I$FFTW_PREFIX/include -L$FFTW_PREFIX/lib -Wl,-rpath,$FFTW_PREFIX/lib -lfftw3f"; else FFTW="-lfftw3f"; fi
$CXX $FLAGS "$HERE/standalone/bench_vs_fftw.cpp" $FFTW -o "$HERE/standalone/bench_vs_fftw"
echo "run: $HERE/standalone/test_accuracy && taskset -c 2 $HERE/standalone/bench_vs_fftw 64 1048576 2"
