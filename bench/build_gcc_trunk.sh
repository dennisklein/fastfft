#!/usr/bin/env bash
# Build GCC trunk (C and C++ only) into a local prefix, for the C++26 std::simd measurements. No distribution
# packages it yet. Works on recent dnf-based (Fedora, RHEL/Alma/Rocky >= 9) and apt-based (Debian >= 12,
# Ubuntu >= 22.04) systems.
#
#   bench/build_gcc_trunk.sh [--prefix DIR] [--ref BRANCH|TAG|COMMIT] [--no-deps] [-j N]
#
# --prefix   install location (default: build-<hostname>/gcc-trunk in this repository; nothing outside it is touched)
# --ref      what to build (default: master); the exact commit is written to <prefix>/gcc-commit.txt
# --no-deps  do not install build dependencies (otherwise dnf/apt via sudo when not root)
#
# Takes ~15 min on 4 cores (no bootstrap) and ~3 GB of disk. Afterwards:  bench/run.sh --gcc-trunk <prefix>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PREFIX="$ROOT/build-$(hostname -s)/gcc-trunk" REF=master DEPS=1 JOBS=$(nproc)
while [ $# -gt 0 ]; do
  case $1 in
    --prefix) PREFIX=$2; shift 2 ;;
    --ref) REF=$2; shift 2 ;;
    --no-deps) DEPS=0; shift ;;
    -j) JOBS=$2; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
PREFIX=$(mkdir -p "$PREFIX" && cd "$PREFIX" && pwd)
WORK="$PREFIX.work"
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO=sudo

if [ "$DEPS" = 1 ]; then
  if command -v dnf >/dev/null; then
    $SUDO dnf install -y gcc gcc-c++ make git flex bison gmp-devel mpfr-devel libmpc-devel zlib-devel diffutils
  elif command -v apt-get >/dev/null; then
    $SUDO apt-get update -q
    $SUDO apt-get install -y -q build-essential git flex bison libgmp-dev libmpfr-dev libmpc-dev zlib1g-dev
  else
    echo "neither dnf nor apt-get found; install a C/C++ compiler, make, git, flex, bison, GMP, MPFR and MPC" \
         "development packages, then rerun with --no-deps" >&2
    exit 1
  fi
fi

mkdir -p "$WORK"
if [ ! -d "$WORK/src/.git" ]; then # shallow clone; the GitHub mirror is usually faster than gcc.gnu.org
  git clone --depth 1 --branch "$REF" https://github.com/gcc-mirror/gcc.git "$WORK/src" 2>/dev/null \
    || git clone --depth 1 --branch "$REF" https://gcc.gnu.org/git/gcc.git "$WORK/src" 2>/dev/null \
    || { git clone https://github.com/gcc-mirror/gcc.git "$WORK/src" && git -C "$WORK/src" checkout "$REF"; }
fi
COMMIT=$(git -C "$WORK/src" rev-parse HEAD)
echo "building GCC $(cat "$WORK/src/gcc/BASE-VER") at $COMMIT into $PREFIX"

mkdir -p "$WORK/obj"
cd "$WORK/obj"
# --disable-bootstrap: one stage, built by the system compiler; fine for a benchmark compiler and 3x faster
[ -f Makefile ] || "$WORK/src/configure" --prefix="$PREFIX" --enable-languages=c,c++ --disable-bootstrap \
  --disable-multilib --disable-nls --disable-libsanitizer --disable-libgomp --disable-libquadmath \
  --disable-libitm --disable-libvtv >"$WORK/configure.log"
make -j"$JOBS" >"$WORK/make.log" 2>&1 || { tail -30 "$WORK/make.log"; exit 1; }
make install >"$WORK/install.log" 2>&1
echo "$COMMIT" >"$PREFIX/gcc-commit.txt"

# smoke test: C++26 <simd> must be available for the target
T=$(mktemp -d)
cat >"$T/t.cpp" <<'CPP'
#include <simd>
int main() { std::simd::vec<float> v(1.f); return int(std::simd::reduce(v)) == int(v.size()) ? 0 : 1; }
CPP
"$PREFIX/bin/g++" -std=c++26 -O2 -march=native -Wl,-rpath,"$PREFIX/lib64" "$T/t.cpp" -o "$T/t" && "$T/t" \
  && echo "OK: $("$PREFIX/bin/g++" --version | head -1), C++26 <simd> works. Next: bench/run.sh --gcc-trunk $PREFIX" \
  || { echo "GCC built, but C++26 <simd> did not compile/run (x86-64 only so far)" >&2; exit 1; }
rm -rf "$T"
echo "the build tree $WORK can be deleted"
