// times fastfft::stdsimd::Plan (build once per width via -DFASTFFT_SIMD_FLOATS=4/8/16 or -march) and,
// with -DWITH_SIMDFFT, gr4's SimdFFT; prints "N ns" per line
#include "FastFFTSimd.hpp"
#ifdef WITH_SIMDFFT
#include <gnuradio-4.0/algorithm/fourier/SimdFFT.hpp>
gr::exception::exception(std::string_view, std::source_location) noexcept {}
const char* gr::exception::what() const noexcept { return "gr::exception"; }
#endif
#include <chrono>
#include <cstdio>
#include <random>
using Cf = std::complex<float>;
template<class F> double timeIt(F&& f){ using clk=std::chrono::steady_clock; int reps=1;
  for(;;){auto t0=clk::now(); for(int i=0;i<reps;i++)f(); double d=std::chrono::duration<double>(clk::now()-t0).count(); if(d>1e-3)break; reps*=2;}
  double best=1e9; for(int k=0;k<31;k++){auto t0=clk::now(); for(int i=0;i<reps;i++)f(); best=std::min(best,std::chrono::duration<double>(clk::now()-t0).count()/reps);} return best;}
int main(){
  std::mt19937 g(1); std::uniform_real_distribution<float> d(-1,1);
  fprintf(stderr,"floats/vector=%zu\n", fastfft::stdsimd::W);
  for (size_t N : {64UL,256UL,1024UL,4096UL,16384UL,65536UL,262144UL,1048576UL}){
    const size_t total=std::max<size_t>(N,1<<16), B=total/N;
    auto* x=(Cf*)std::aligned_alloc(64,total*8); auto* y=(Cf*)std::aligned_alloc(64,total*8);
    for(size_t i=0;i<total;i++) x[i]={d(g),d(g)};
#ifdef WITH_SIMDFFT
    using namespace gr::algorithm; SimdFFT<float, Transform::Complex> f; f.resize(N);
    double t=timeIt([&]{for(size_t b=0;b<B;b++) f.template transform<Direction::Forward, Order::Ordered>(std::span<const float>((float*)(x+b*N),2*N), std::span<float>((float*)(y+b*N),2*N));})/B;
#else
    fastfft::stdsimd::Plan p(N);
    double t=timeIt([&]{for(size_t b=0;b<B;b++) p.forward(x+b*N,y+b*N);})/B;
#endif
    printf("%zu %.0f\n",N,t*1e9); free(x); free(y);
  }
}
