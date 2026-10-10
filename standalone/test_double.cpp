// double precision: native_simd port (Simd<double>) vs gr4 SimdFFT<double> (-DWITH_SIMDFFT) vs FFTW (-DWITH_FFTW)
#include "FastFFTSimd.hpp"
#ifdef WITH_SIMDFFT
#include <gnuradio-4.0/algorithm/fourier/SimdFFT.hpp>
gr::exception::exception(std::string_view, std::source_location) noexcept {}
const char* gr::exception::what() const noexcept { return "gr::exception"; }
#endif
#ifdef WITH_FFTW
#include <fftw3.h>
#endif
#include <chrono>
#include <cstdio>
#include <random>
using Cd = std::complex<double>;
using Cl = std::complex<long double>;
void refFFT(std::vector<Cl>& a) { size_t n = a.size(); if (n == 1) return;
    std::vector<Cl> e(n/2), o(n/2); for (size_t i=0;i<n/2;i++){e[i]=a[2*i];o[i]=a[2*i+1];}
    refFFT(e); refFFT(o);
    for (size_t k=0;k<n/2;k++){ Cl w=std::polar(1.0L,-2*std::numbers::pi_v<long double>*(long double)k/(long double)n)*o[k]; a[k]=e[k]+w; a[k+n/2]=e[k]-w;} }
template<class F> double timeIt(F&& f){ using clk=std::chrono::steady_clock; int reps=1;
  for(;;){auto t0=clk::now(); for(int i=0;i<reps;i++)f(); double d=std::chrono::duration<double>(clk::now()-t0).count(); if(d>1e-3)break; reps*=2;}
  double best=1e9; for(int k=0;k<31;k++){auto t0=clk::now(); for(int i=0;i<reps;i++)f(); best=std::min(best,std::chrono::duration<double>(clk::now()-t0).count()/reps);} return best;}
int main(){
  std::mt19937 g(1); std::uniform_real_distribution<double> d(-1,1); double worst=0;
  printf("doubles/vector=%zu\n%8s %10s %10s %10s | %9s %9s\n", fastfft::stdsimd::Simd<double>::W, "N","ns simd","ns SimdFFT","ns fftw","err simd","err SimdFFT");
  for (size_t N=64; N<=(1u<<20); N*=4){
    const size_t total=std::max<size_t>(N,1<<15), B=total/N;
    auto* x=(Cd*)std::aligned_alloc(64,total*16); auto* y=(Cd*)std::aligned_alloc(64,total*16);
    for(size_t i=0;i<total;i++) x[i]={d(g),d(g)};
    std::vector<Cl> r(x,x+N); refFFT(r); long double mx=0; for(auto& v:r) mx=std::max(mx,std::abs(v));
    auto err=[&](const Cd* o){ long double e=0; for(size_t i=0;i<N;i++) e=std::max(e,std::abs(Cl(o[i])-r[i])/mx); return double(e); };
    fastfft::stdsimd::PlanD p(N); p.forward(x,y); double e1=err(y); worst=std::max(worst,e1);
    double ts=timeIt([&]{for(size_t b=0;b<B;b++) p.forward(x+b*N,y+b*N);})/B;
    double tg=0, e2=0, tf=0;
#ifdef WITH_SIMDFFT
    { using namespace gr::algorithm; SimdFFT<double, Transform::Complex> f; f.resize(N);
      f.template transform<Direction::Forward, Order::Ordered>(std::span<const double>((double*)x,2*N), std::span<double>((double*)y,2*N)); e2=err(y);
      tg=timeIt([&]{for(size_t b=0;b<B;b++) f.template transform<Direction::Forward, Order::Ordered>(std::span<const double>((double*)(x+b*N),2*N), std::span<double>((double*)(y+b*N),2*N));})/B; }
#endif
#ifdef WITH_FFTW
    { fftw_plan fp=fftw_plan_dft_1d(int(N),(fftw_complex*)x,(fftw_complex*)y,FFTW_FORWARD,FFTW_MEASURE); for(size_t i=0;i<total;i++) x[i]={d(g),d(g)};
      tf=timeIt([&]{for(size_t b=0;b<B;b++) fftw_execute_dft(fp,(fftw_complex*)(x+b*N),(fftw_complex*)(y+b*N));})/B; fftw_destroy_plan(fp); }
#endif
    printf("%8zu %10.0f %10.0f %10.0f | %9.1e %9.1e\n",N,ts*1e9,tg*1e9,tf*1e9,e1,e2);
    free(x); free(y);
  }
  printf("worst=%.2e %s\n",worst,worst<1e-13?"OK":"FAIL");
}
