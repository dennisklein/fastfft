#!/usr/bin/env bash
# Reproduce the paper's measurements on this Linux host.
#
#   bench/run.sh [--fftw build|system|none] [--runs 3] [--samples 40] [--cpu N] [--cxx g++] [--widths "16 32"]
#                [--libcxx clang++] [--gcc-trunk PREFIX]
#
# Builds fftbench (fastfft, gnuradio4 SimdFFT, FFTW) with -O3 -march=native, records the host configuration, runs the
# benchmark --runs times as separate processes pinned to one core, sweeps narrower simd widths for fastfft, and writes
# everything to results/<hostname>-<date>/ followed by the statistical summary (summary.md). --libcxx additionally
# builds with that clang and libc++, where vir-simd emulates simd with arrays, and times fastfft and SimdFFT there.
# --gcc-trunk (a prefix from bench/build_gcc_trunk.sh) adds the GCC trunk / C++26 builds: label gcc17 (all
# implementations, plus fastfft::cx on std::simd::vec<std::complex<T>>), gcc17-exp and gcc17-cxbc (see below).
#
# Needs: cmake >= 3.24, git, python3, a C++23 compiler with libstdc++ >= 14 (GCC >= 14, or clang with libstdc++),
# util-linux (taskset, lscpu); for --fftw build also make and a C compiler. Default --fftw build compiles FFTW 3.3.10
# with the SIMD codelets of this CPU, as distribution packages often lack AVX-512.
#
# SPDX-FileCopyrightText: 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH, Dennis Klein
# SPDX-License-Identifier: LGPL-3.0-or-later WITH LGPL-3.0-linking-exception
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FFTW=build RUNS=3 SAMPLES=40 CPU="" WIDTHS=auto LIBCXX="" TRUNK="" EXTRA=()
CXX=${CXX:-$(command -v g++-15 || command -v g++-14 || command -v g++)}   # gnuradio4 headers need GCC >= 14
while [ $# -gt 0 ]; do
  case $1 in
    --fftw) FFTW=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --samples) SAMPLES=$2; shift 2 ;;
    --cpu) CPU=$2; shift 2 ;;
    --cxx) CXX=$2; shift 2 ;;
    --widths) WIDTHS=$2; shift 2 ;;
    --libcxx) LIBCXX=$2; shift 2 ;;
    --gcc-trunk) TRUNK=$(cd "$2" && pwd); shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;   # passed to fftbench, e.g. -- --max-log2 16
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

HOST=$(hostname -s)
OUT="$ROOT/results/$HOST-$(date +%Y%m%d-%H%M)"
BUILD="$ROOT/build-$HOST"
mkdir -p "$OUT" "$BUILD"
ARCH=$(uname -m)
FLAGS=$(grep -m1 -E '^(flags|Features)' /proc/cpuinfo || true)
has() { [[ " $FLAGS " == *" $1 "* ]]; }

# --- FFTW --------------------------------------------------------------------------------------------------------
FFTW_ARGS=(-DFFTBENCH_FFTW=OFF)
if [ "$FFTW" = system ]; then
  FFTW_ARGS=(-DFFTBENCH_FFTW=ON)
elif [ "$FFTW" = build ]; then
  P="$BUILD/fftw"
  if [ ! -f "$P/lib/libfftw3f.a" ] || [ ! -f "$P/lib/libfftw3.a" ]; then
    S="$BUILD/fftw-src"; mkdir -p "$S"
    T="$BUILD/fftw-3.3.10.tar.gz"
    [ -s "$T" ] || curl -fsSL -o "$T" https://www.fftw.org/fftw-3.3.10.tar.gz \
      || curl -fsSL -o "$T" http://archive.ubuntu.com/ubuntu/pool/main/f/fftw3/fftw3_3.3.10.orig.tar.gz
    # fftw.org release tarball, or the (repacked) Ubuntu orig tarball of the same release
    sha256sum "$T" | grep -qE '^(56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467|5499874012aa6ea6c597b8c3807494e271ab2426a889e95d5cc4d154b825e8a0) ' \
      || { echo "unexpected checksum of $T" >&2; exit 1; }
    tar -xzf "$T" -C "$S" --strip-components=1
    SIMD=()
    case $ARCH in
      x86_64)
        has sse2 && SIMD+=(--enable-sse2); has avx && SIMD+=(--enable-avx); has avx2 && SIMD+=(--enable-avx2)
        has fma && SIMD+=(--enable-fma); has avx512f && SIMD+=(--enable-avx512) ;;
      aarch64) SIMD+=(--enable-neon) ;;   # FFTW 3.3.10 NEON codelets cover single and double precision
    esac
    for prec in "" --enable-float; do
      (cd "$S" && make distclean >/dev/null 2>&1 || true
       cd "$S" && CFLAGS="-O3 -march=native" ./configure -q --prefix="$P" --disable-fortran $prec "${SIMD[@]}" \
         && make -s -j"$(nproc)" && make -s install) >"$BUILD/fftw-build${prec:+-float}.log" 2>&1
    done
    echo "FFTW 3.3.10 built with: ${SIMD[*]:-no simd}" | tee "$BUILD/fftw-config.txt"
  fi
  FFTW_ARGS=(-DFFTBENCH_FFTW=ON -DFFTBENCH_FFTW_ROOT="$P")
fi

# --- build -------------------------------------------------------------------------------------------------------
if [ "$WIDTHS" = auto ]; then # every power-of-two width from 16 bytes up to, excluding, the native one
  case $ARCH in
    x86_64) WIDTHS=""; has avx && WIDTHS="16"; has avx512f && WIDTHS="16 32" ;;
    *) WIDTHS="" ;;
  esac
fi
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" "${FFTW_ARGS[@]}" \
      -DFFTBENCH_WIDTHS="${WIDTHS// /;}" >"$OUT/cmake.log"
cmake --build "$BUILD" -j"$(nproc)" >>"$OUT/cmake.log"
if [ -n "$LIBCXX" ]; then
  cmake -S "$ROOT" -B "$BUILD-libcxx" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$LIBCXX" -DCMAKE_CXX_FLAGS=-stdlib=libc++ \
        -DFFTBENCH_FFTW=OFF -DFETCHCONTENT_SOURCE_DIR_GR4="$BUILD/_deps/gr4-src" \
        -DFETCHCONTENT_SOURCE_DIR_VIRSIMD="$BUILD/_deps/virsimd-src" >>"$OUT/cmake.log"
  cmake --build "$BUILD-libcxx" -j"$(nproc)" --target fftbench >>"$OUT/cmake.log"
fi

if [ -n "$TRUNK" ]; then
  # gcc17:       the full set (fastfft on C++26 std::simd, fastfft::cx, SimdFFT, FFTW) built with GCC trunk, -std=c++26
  # gcc17-exp:   fastfft on <experimental/simd> with the same compiler (separates the API from the compiler)
  # gcc17-cxbc:  fastfft::cx with zip/broadcast via bit_cast to a real vector (attributes the cx gap; not portable)
  for v in gcc17: gcc17-exp:-DFASTFFT_NO_STD_SIMD gcc17-cxbc:-DFASTFFT_CX_EXPERIMENT_BITCAST; do
    l=${v%%:*} def=${v#*:}
    cmake -S "$ROOT" -B "$BUILD-$l" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$TRUNK/bin/g++" \
          -DFFTBENCH_STD_FLAG=-std=c++26 -DFFTBENCH_CX=ON "${FFTW_ARGS[@]}" \
          -DFETCHCONTENT_SOURCE_DIR_GR4="$BUILD/_deps/gr4-src" -DFETCHCONTENT_SOURCE_DIR_VIRSIMD="$BUILD/_deps/virsimd-src" \
          -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$TRUNK/lib64" -DCMAKE_CXX_FLAGS="$def" >>"$OUT/cmake.log"
    cmake --build "$BUILD-$l" -j"$(nproc)" --target fftbench >>"$OUT/cmake.log"
  done
  echo "check: accuracy of the GCC trunk build"
  "$BUILD-gcc17/fftbench" --samples 1 --min-sample-ms 0.1 --check --label gcc17 --out "$OUT/check-gcc17.csv" 2>>"$OUT/check.log"
fi

# --- host record -------------------------------------------------------------------------------------------------
{
  echo "date: $(date -Iseconds)"; echo "uname: $(uname -a)"
  echo "compiler: $("$CXX" --version | head -1)"; echo "cmake: $(cmake --version | head -1)"
  [ -n "$LIBCXX" ] && echo "libc++ build: $("$LIBCXX" --version | head -1)"
  [ -n "$TRUNK" ] && echo "gcc trunk: $("$TRUNK/bin/g++" --version | head -1) $(cat "$TRUNK/gcc-commit.txt" 2>/dev/null)"
  echo "fastfft: $(git -C "$ROOT" describe --always --dirty 2>/dev/null || echo unknown)"
  [ -f "$BUILD/fftw-config.txt" ] && cat "$BUILD/fftw-config.txt"
  echo "flags: $(grep -m1 CMAKE_CXX_FLAGS_RELEASE: "$BUILD/CMakeCache.txt") -O3 -march=native -ffp-contract=fast"
  gov=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c | xargs || true)
  echo "governor: ${gov:-n/a}"
  echo "turbo: no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo n/a)" \
       "boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a)"
  echo "smt: $(cat /sys/devices/system/cpu/smt/active 2>/dev/null || echo n/a)"
  echo "load: $(cat /proc/loadavg)"
  echo; lscpu
} >"$OUT/host.txt" 2>&1

# --- measure -----------------------------------------------------------------------------------------------------
[ -n "$CPU" ] || CPU=$(( $(nproc) - 1 ))   # pin to one core; the last one is usually least disturbed
echo "pinned to cpu $CPU" >>"$OUT/host.txt"
echo "check: accuracy of all implementations"
"$BUILD/fftbench" --samples 1 --min-sample-ms 0.1 --check --out "$OUT/check.csv" 2>"$OUT/check.log"
for r in $(seq 1 "$RUNS"); do
  echo "run $r/$RUNS: all implementations, float + double, N = 2^6..2^20"
  taskset -c "$CPU" "$BUILD/fftbench" --samples "$SAMPLES" --out "$OUT/raw-$r.csv" "${EXTRA[@]}" 2>>"$OUT/run.log"
  for w in $WIDTHS; do
    echo "run $r/$RUNS: fastfft forced to $w-byte vectors"
    taskset -c "$CPU" "$BUILD/fftbench_w$w" --samples "$SAMPLES" --impls fastfft --precision float \
      --out "$OUT/width-raw-w$w-$r.csv" "${EXTRA[@]}" 2>>"$OUT/run.log"
  done
  if [ -n "$LIBCXX" ]; then
    echo "run $r/$RUNS: fastfft and SimdFFT built with $LIBCXX + libc++"
    taskset -c "$CPU" "$BUILD-libcxx/fftbench" --samples "$SAMPLES" --impls fastfft,gr4-SimdFFT --precision float \
      --label libcxx --out "$OUT/libcxx-raw-$r.csv" "${EXTRA[@]}" 2>>"$OUT/run.log"
  fi
  if [ -n "$TRUNK" ]; then
    echo "run $r/$RUNS: GCC trunk, C++26: all implementations"
    taskset -c "$CPU" "$BUILD-gcc17/fftbench" --samples "$SAMPLES" --label gcc17 \
      --out "$OUT/gcc17-raw-$r.csv" "${EXTRA[@]}" 2>>"$OUT/run.log"
    echo "run $r/$RUNS: GCC trunk, experimental simd / cx bit_cast experiment"
    taskset -c "$CPU" "$BUILD-gcc17-exp/fftbench" --samples "$SAMPLES" --label gcc17-exp --impls fastfft \
      --out "$OUT/gcc17-exp-raw-$r.csv" "${EXTRA[@]}" 2>>"$OUT/run.log"
    taskset -c "$CPU" "$BUILD-gcc17-cxbc/fftbench" --samples "$SAMPLES" --label gcc17-cxbc --impls fastfft,fastfft-cx \
      --out "$OUT/gcc17-cxbc-raw-$r.csv" "${EXTRA[@]}" 2>>"$OUT/run.log"
  fi
done
python3 "$ROOT/bench/analyze.py" "$OUT"
echo "results in $OUT"
