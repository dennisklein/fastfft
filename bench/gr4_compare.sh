#!/usr/bin/env bash
# Effect of gr4/0001-fastfft.patch on gnuradio4 itself, measured with gnuradio4's own benchmark facility.
#
#   bench/gr4_compare.sh [--reps 5] [--cpu N]          (CC/CXX select the compiler, default: newest gcc/g++)
#
# Checks out gnuradio4 at the pinned revision twice -- unpatched ("base") and with 0001-fastfft.patch ("patch") --,
# adds the stream-block benchmark (0002-bm_fft_stream.patch) to both, builds bm_fft_stream, bm_fft_block and the FFT
# unit tests with gnuradio4's Release settings plus -march=native, runs the unit tests of the patched tree, then runs
# both benchmarks --reps times, alternating base and patch, pinned to one core. Output: results/<host>-<date>-gr4/.
# Needs what gnuradio4 needs (cmake, ninja, GCC >= 14). Use GCC: gnuradio4's clang build uses libc++, where vir-simd
# emulates simd with arrays and the patch therefore keeps SimdFFT (clang 20 with libstdc++ 14.2 does not build gr4).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REV=af6223528252b6b98503bf92623eb9054ea44c32
REPS=5 CPU=""
while [ $# -gt 0 ]; do
  case $1 in
    --reps) REPS=$2; shift 2 ;;
    --cpu) CPU=$2; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
CXX=${CXX:-$(command -v g++-15 || command -v g++-14 || command -v g++)}
CC=${CC:-${CXX/g++/gcc}}
HOST=$(hostname -s)
OUT="$ROOT/results/$HOST-$(date +%Y%m%d-%H%M)-gr4"
W="$ROOT/build-$HOST/gr4"
mkdir -p "$OUT" "$W"
[ -n "$CPU" ] || CPU=$(( $(nproc) - 1 ))

if [ ! -e "$W/base/.git" ]; then
  git clone -q https://github.com/fair-acc/gnuradio4.git "$W/base"
  git -C "$W/base" checkout -q "$REV"
  git -C "$W/base" worktree add -q "$W/patch" "$REV"
  git -C "$W/base" apply "$ROOT/gr4/0002-bm_fft_stream.patch"
  git -C "$W/patch" apply "$ROOT/gr4/0001-fastfft.patch" "$ROOT/gr4/0002-bm_fft_stream.patch"
fi
TARGETS=(bm_fft_stream bm_fft_block qa_FFT qa_algorithm_fourier)
for v in base patch; do
  cmake -S "$W/$v" -B "$W/$v/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CXX_FLAGS=-march=native \
        -DGR_ENABLE_HTTP=OFF -DENABLE_TESTING=ON -DENABLE_BENCHMARKS=ON -DENABLE_EXAMPLES=OFF >"$OUT/cmake-$v.log"
  echo "building $v (this takes a while)"
  cmake --build "$W/$v/build" --target "${TARGETS[@]}" >>"$OUT/cmake-$v.log"
done
{ echo "date: $(date -Iseconds)"; echo "uname: $(uname -a)"; echo "compiler: $("$CXX" --version | head -1)"
  echo "gnuradio4: $REV"; echo "pinned to cpu $CPU"; echo; lscpu; } >"$OUT/host.txt"

echo "unit tests (patched tree)"
for t in blocks/fourier/test/qa_FFT algorithm/test/qa_algorithm_fourier; do
  "$W/patch/build/$t" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^Suite|FAIL" | tee -a "$OUT/tests.txt"
done
for r in $(seq 1 "$REPS"); do
  for v in base patch; do
    echo "rep $r/$REPS: $v"
    for b in bm_fft_stream bm_fft_block; do
      BM_DIGITS=4 taskset -c "$CPU" "$W/$v/build/blocks/fourier/benchmarks/$b" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' >"$OUT/$b-$v-$r.txt"
    done
  done
done
python3 "$ROOT/bench/gr4_analyze.py" "$OUT"
