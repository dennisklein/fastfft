// accuracy + timing of the native_simd port vs. the hand-written AVX-512 kernel (and SimdFFT if GR4 headers given)
#include "FastFFTSimd.hpp"
#ifdef __AVX512F__
#include "FastFFT.hpp"
#endif
#include <chrono>
#include <cstdio>
#include <random>
using cd = std::complex<double>;
using Cf = std::complex<float>;
void refFFT(std::vector<cd>& a) { size_t n = a.size(); if (n == 1) return;
    std::vector<cd> e(n/2), o(n/2); for (size_t i=0;i<n/2;i++){e[i]=a[2*i];o[i]=a[2*i+1];}
    refFFT(e); refFFT(o);
    for (size_t k=0;k<n/2;k++){ cd w=std::polar(1.0,-2*std::numbers::pi*double(k)/double(n))*o[k]; a[k]=e[k]+w; a[k+n/2]=e[k]-w;} }
template<class F> double timeIt(F&& f){ using clk=std::chrono::steady_clock; int reps=1;
  for(;;){auto t0=clk::now(); for(int i=0;i<reps;i++)f(); double d=std::chrono::duration<double>(clk::now()-t0).count(); if(d>1e-3)break; reps*=2;}
  double best=1e9; for(int k=0;k<31;k++){auto t0=clk::now(); for(int i=0;i<reps;i++)f(); best=std::min(best,std::chrono::duration<double>(clk::now()-t0).count()/reps);} return best;}
int main(int argc, char**){
  std::mt19937 g(1); std::uniform_real_distribution<float> d(-1,1); double worst=0;
  printf("native_simd<float>::size() = %zu\n", fastfft::stdsimd::W);
  printf("%8s %10s %10s %8s | %9s\n","N","ns simd","ns avx512","ratio","err simd");
  for (size_t N=8; N<=(1u<<20); N*=2){
    const size_t total=std::max<size_t>(N,1<<16), B=total/N;
    auto* x=(Cf*)std::aligned_alloc(64,total*8); auto* y=(Cf*)std::aligned_alloc(64,total*8);
    for(size_t i=0;i<total;i++) x[i]={d(g),d(g)};
    std::vector<cd> r(x,x+N); refFFT(r); double mx=0; for(auto& v:r) mx=std::max(mx,std::abs(v));
    fastfft::stdsimd::Plan sp(N); sp.forward(x,y);
    double e=0; for(size_t i=0;i<N;i++) e=std::max(e,std::abs(cd(y[i])-r[i])/mx); worst=std::max(worst,e);
    std::vector<Cf> xx(x,x+N); sp.forward(xx.data(),xx.data()); for(size_t i=0;i<N;i++) worst=std::max(worst,std::abs(cd(xx[i])-r[i])/mx);
    double ts = argc>1 ? 0 : timeIt([&]{for(size_t b=0;b<B;b++) sp.forward(x+b*N,y+b*N);})/B;
    double th = 0;
#ifdef __AVX512F__
    fastfft::Plan hp(N); if(argc<=1) th=timeIt([&]{for(size_t b=0;b<B;b++) hp.forward(x+b*N,y+b*N);})/B;
#endif
    if (N>=64) printf("%8zu %10.0f %10.0f %8.2f | %9.1e\n",N,ts*1e9,th*1e9,th>0?ts/th:0.0,e);
    free(x); free(y);
  }
  printf("worst=%.2e %s\n",worst,worst<1e-6?"OK":"FAIL");
}
