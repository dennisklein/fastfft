#!/usr/bin/env bash
# Builds gr4/bm_fft_fast inside a gnuradio4 checkout. Needs clang >= 20 + libc++ (or gcc >= 14.3), cmake >= 3.27, ninja.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
GR4=${GR4:-$HERE/gnuradio4}
GR4_REV=${GR4_REV:-af6223528252b6b98503bf92623eb9054ea44c32}
CXX=${CXX:-clang++-20}
CC=${CC:-clang-20}
if [ ! -d "$GR4" ]; then
  git clone https://github.com/fair-acc/gnuradio4 "$GR4"
  git -C "$GR4" checkout "$GR4_REV"
fi
BM=$GR4/blocks/fourier/benchmarks
mkdir -p "$BM/fastfft"
cp "$HERE"/fastfft/*.hpp "$BM/fastfft/"
cp "$HERE"/gr4/*.cpp "$HERE"/gr4/*.hpp "$BM/"
grep -q bm_fft_fast "$BM/CMakeLists.txt" || cat >> "$BM/CMakeLists.txt" <<'CM'

add_gr_benchmark(bm_fft_fast)
add_gr_benchmark(bm_fft_spectrum)
add_gr_benchmark(bm_fft_spectrum_fast)
target_compile_options(bm_fft_spectrum_fast PRIVATE -O3 -march=native -mtune=native)
target_compile_options(bm_fft_spectrum PRIVATE -O3 -march=native -mtune=native)
target_compile_options(bm_fft_fast PRIVATE -O3 -march=native -mtune=native)
target_link_options(bm_fft_fast PRIVATE -march=native)
CM
cmake -S "$GR4" -B "$GR4/build" -G Ninja -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-march=native" -DENABLE_TESTING=OFF -DENABLE_EXAMPLES=OFF -DENABLE_BENCHMARKS=ON -DWARNINGS_AS_ERRORS=OFF \
  -DGR_ENABLE_BLOCK_REGISTRY=OFF -DINTERNAL_ENABLE_BLOCK_PLUGINS=OFF -DGR_ENABLE_HTTP=OFF
ninja -C "$GR4/build" bm_fft_fast bm_fft_spectrum bm_fft_spectrum_fast
echo "run: taskset -c 2 $GR4/build/blocks/fourier/benchmarks/bm_fft_fast"
