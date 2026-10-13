#!/usr/bin/env bash
# Build gnuradio4 with gr4/patches applied, run the StockhamFFT tests and bm_fft per -march target, and store the results.
#   tools/measure.sh [OUT]          OUT defaults to results/<host>-<yyyymmdd>
# Environment (all optional):
#   GR4_DIR        gnuradio4 checkout to use; default build/gnuradio4, created by gr4/apply.sh
#   TARGETS        -march values to measure; default: native, plus x86-64 and x86-64-v3 on x86-64 hosts
#   CXX            compiler; its libstdc++ directory goes into the rpath
#   CXX_STANDARD   23 (default) or 26 for the std::simd preview path (GCC trunk, CMake >= 3.30)
#   CPU            core to pin the benchmark to with taskset; default 2 (or 0 on hosts with fewer cores)
#   BM_FFT_ROUNDS, BM_FFT_MAX_LOG2, BM_FFT_MIN_SAMPLE_MS   passed to bm_fft; defaults 40, 20, 5
#   JOBS           parallel build jobs; default nproc
# SPDX-FileCopyrightText: 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later WITH LGPL-3.0-linking-exception
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HOST=$(hostname -s 2>/dev/null || echo host)
OUT=$(realpath -m "${1:-$ROOT/results/$HOST-$(date +%Y%m%d)}")
GR4_DIR=$(realpath -m "${GR4_DIR:-$ROOT/build/gnuradio4}")
CXX_STANDARD=${CXX_STANDARD:-23}
JOBS=${JOBS:-$(nproc)}
export BM_FFT_ROUNDS=${BM_FFT_ROUNDS:-40} BM_FFT_MAX_LOG2=${BM_FFT_MAX_LOG2:-20} BM_FFT_MIN_SAMPLE_MS=${BM_FFT_MIN_SAMPLE_MS:-5}

if [ -z "${TARGETS:-}" ]; then
    TARGETS=native
    [ "$(uname -m)" = x86_64 ] && TARGETS="x86-64 x86-64-v3 native"
fi
CPU=${CPU:-$([ "$(nproc)" -gt 2 ] && echo 2 || echo 0)}
PIN=()
command -v taskset >/dev/null && PIN=(taskset -c "$CPU")

[ -d "$GR4_DIR/.git" ] || "$ROOT/gr4/apply.sh" "$GR4_DIR"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD="$CXX_STANDARD"
    -DENABLE_EXAMPLES=OFF -DGR_ENABLE_BLOCK_REGISTRY=OFF -DINTERNAL_ENABLE_BLOCK_PLUGINS=OFF -DGR_ENABLE_HTTP=OFF)
[ "$CXX_STANDARD" -ge 26 ] && CMAKE_ARGS+=(-DWARNINGS_AS_ERRORS=OFF) # gr4 core still warns under C++26
if [ -n "${CXX:-}" ]; then
    LIBDIR=$(dirname "$("$CXX" -print-file-name=libstdc++.so)")
    CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$LIBDIR")
fi

mkdir -p "$OUT"
{
    echo "date: $(date -Iseconds)"
    echo "gnuradio4: $(git -C "$GR4_DIR" describe --always --dirty) ($(git -C "$GR4_DIR" log -1 --format=%s))"
    echo "compiler: $("${CXX:-c++}" --version | head -1)"
    echo "c++ standard: $CXX_STANDARD"
    echo "targets: $TARGETS"
    echo "pinned: ${PIN[*]:-no}"
    echo "kernel: $(uname -srm)"
    echo
    lscpu 2>/dev/null || sysctl -a 2>/dev/null | grep -E '^(machdep\.cpu|hw\.)' || true
} > "$OUT/host.txt"

for target in $TARGETS; do
    build="$ROOT/build/build-$target-cxx$CXX_STANDARD"
    echo "== $target: configuring and building in $build"
    cmake -S "$GR4_DIR" -B "$build" "${CMAKE_ARGS[@]}" -DCMAKE_CXX_FLAGS="-march=$target" > "$OUT/cmake-$target.log" 2>&1 \
        || { echo "configure failed, see $OUT/cmake-$target.log"; exit 1; }
    cmake --build "$build" -j "$JOBS" --target qa_StockhamFFT qa_algorithm_fourier bm_fft > "$OUT/build-$target.log" 2>&1 \
        || { echo "build failed, see $OUT/build-$target.log"; exit 1; }
    for qa in qa_StockhamFFT qa_algorithm_fourier; do
        "$(find "$build" -name "$qa" -type f -executable | head -1)" > "$OUT/$qa-$target.log" 2>&1 \
            || { echo "$qa failed for $target, see $OUT/$qa-$target.log"; exit 1; }
    done
    echo "== $target: running bm_fft (BM_FFT_ROUNDS=$BM_FFT_ROUNDS BM_FFT_MAX_LOG2=$BM_FFT_MAX_LOG2 BM_FFT_MIN_SAMPLE_MS=$BM_FFT_MIN_SAMPLE_MS)"
    "${PIN[@]}" "$(find "$build" -name bm_fft -type f -executable | head -1)" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$OUT/bm-$target.txt" || true
    if grep -q 'relative L2 error' "$OUT/bm-$target.txt"; then
        echo "!! $target: some cases failed the accuracy check and were left out of timing and routes:"
        grep 'relative L2 error' "$OUT/bm-$target.txt" | sort -u
    fi
    grep -E '^(// (Float|Double)|inline constexpr std::array|    FFTRoute|\};)' "$OUT/bm-$target.txt" > "$OUT/routes-$target.txt" || true
done

python3 "$ROOT/tools/summarise.py" "$OUT"/bm-*.txt > "$OUT/summary.md"
echo "== results in $OUT: summary.md, routes-<target>.txt (paste into FFTRoutes.hpp), bm-<target>.txt"
