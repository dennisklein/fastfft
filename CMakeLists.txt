# fastfft: header-only power-of-two complex FFT on std::simd, plus the benchmark used in the paper.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release [-DFFTBENCH_FFTW_ROOT=/path/to/fftw]
#   cmake --build build
#
# Options
#   FFTBENCH_SIMDFFT    ON   also time gnuradio4's SimdFFT (fetches gnuradio4 + vir-simd sources, headers only)
#   FFTBENCH_FFTW       AUTO also time FFTW 3 (ON: required, OFF: skip, AUTO: if found); see FFTBENCH_FFTW_ROOT
#   FFTBENCH_WIDTHS     ""   extra fastfft builds with a forced vector size in bytes, e.g. "16;32" (width sweep)
cmake_minimum_required(VERSION 3.24)
project(fastfft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FFTBENCH_SIMDFFT "time gnuradio4 SimdFFT" ON)
set(FFTBENCH_FFTW AUTO CACHE STRING "time FFTW: ON, OFF or AUTO")
set(FFTBENCH_FFTW_ROOT "" CACHE PATH "prefix of an FFTW 3 install with both libfftw3 and libfftw3f")
set(FFTBENCH_WIDTHS "" CACHE STRING "extra forced simd widths in bytes for fastfft, e.g. 16;32")
set(FFTBENCH_ARCH_FLAGS "-march=native" CACHE STRING "target flags (the kernel picks the native simd width)")

add_library(fastfft INTERFACE)
target_include_directories(fastfft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# the kernel writes a*b+c and relies on contraction to FMA (libstdc++'s stdx::fma is an out-of-line call)
target_compile_options(fastfft INTERFACE -ffp-contract=fast)

include(FetchContent)
set(defs "")
set(incs "")
set(libs "")
if(FFTBENCH_SIMDFFT)
  # pinned to the revision the paper measured; only headers are used, nothing of gnuradio4 is configured or built
  FetchContent_Declare(gr4 GIT_REPOSITORY https://github.com/fair-acc/gnuradio4.git
                           GIT_TAG af6223528252b6b98503bf92623eb9054ea44c32 SOURCE_SUBDIR _none_)
  FetchContent_Declare(virsimd GIT_REPOSITORY https://github.com/mattkretz/vir-simd.git GIT_TAG v0.4.4
                               GIT_SHALLOW ON SOURCE_SUBDIR _none_)
  FetchContent_MakeAvailable(gr4 virsimd)
  list(APPEND defs FFTBENCH_WITH_SIMDFFT)
  list(APPEND incs ${gr4_SOURCE_DIR}/algorithm/include ${gr4_SOURCE_DIR}/core/include ${gr4_SOURCE_DIR}/meta/include
       ${gr4_SOURCE_DIR}/third_party/magic_enum ${virsimd_SOURCE_DIR})
endif()

if(NOT FFTBENCH_FFTW STREQUAL "OFF")
  find_path(FFTW_INCLUDE fftw3.h HINTS ${FFTBENCH_FFTW_ROOT}/include)
  find_library(FFTW_D fftw3 HINTS ${FFTBENCH_FFTW_ROOT}/lib)
  find_library(FFTW_F fftw3f HINTS ${FFTBENCH_FFTW_ROOT}/lib)
  if(FFTW_INCLUDE AND FFTW_D AND FFTW_F)
    message(STATUS "FFTW: ${FFTW_F} ${FFTW_D}")
    list(APPEND defs FFTBENCH_WITH_FFTW)
    list(APPEND incs ${FFTW_INCLUDE})
    list(APPEND libs ${FFTW_F} ${FFTW_D})
  elseif(FFTBENCH_FFTW STREQUAL "ON")
    message(FATAL_ERROR "FFTW requested but libfftw3/libfftw3f not found (set FFTBENCH_FFTW_ROOT)")
  else()
    message(STATUS "FFTW: not found, skipped")
  endif()
endif()

function(add_bench name)
  add_executable(${name} bench/fftbench.cpp)
  target_link_libraries(${name} PRIVATE fastfft ${libs})
  target_compile_definitions(${name} PRIVATE ${defs} ${ARGN})
  target_include_directories(${name} SYSTEM PRIVATE ${incs})
  separate_arguments(arch UNIX_COMMAND "${FFTBENCH_ARCH_FLAGS}")
  target_compile_options(${name} PRIVATE -O3 ${arch} $<$<CXX_COMPILER_ID:GNU>:-Wno-interference-size>)
endfunction()

add_bench(fftbench)
foreach(w IN LISTS FFTBENCH_WIDTHS)
  add_bench(fftbench_w${w} FASTFFT_SIMD_BYTES=${w})
endforeach()

enable_testing()
add_test(NAME accuracy COMMAND fftbench --samples 1 --min-sample-ms 0.1 --max-log2 14 --check --out ${CMAKE_BINARY_DIR}/test.csv)
