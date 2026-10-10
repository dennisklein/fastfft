#include "FastFFT.hpp"
#include <fftw3.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
using C = std::complex<float>;
template<class F> double timeIt(F&& f){ // min over 41 runs, each run >= ~1ms
  using clk=std::chrono::steady_clock; int reps=1;
  for(;;){auto t0=clk::now(); for(int i=0;i<reps;i++)f(); double d=std::chrono::duration<double>(clk::now()-t0).count(); if(d>1e-3)break; reps*=2;}
  double best=1e9; for(int k=0;k<41;k++){auto t0=clk::now(); for(int i=0;i<reps;i++)f(); best=std::min(best,std::chrono::duration<double>(clk::now()-t0).count()/reps);} return best;
}
int main(int argc,char**argv){
  std::mt19937 g(1); std::uniform_real_distribution<float> d(-1,1);
  size_t Nmin = argc>1? atoi(argv[1]):64, Nmax = argc>2? atoi(argv[2]):(1u<<20), step = argc>3? atoi(argv[3]):4;
  printf("%8s %10s %10s %10s %10s | %6s %6s %6s\n","N","ns stock","ns 4step","ns plan","ns fftw","GFst","GF4s","fftw/best");
  for(size_t N=Nmin; N<=Nmax; N*=step){
    const size_t total=std::max<size_t>(N,1<<16), B=total/N;
    C* x=(C*)std::aligned_alloc(64,total*8); C* y=(C*)std::aligned_alloc(64,total*8); C* z=(C*)std::aligned_alloc(64,total*8);
    for(size_t i=0;i<total;i++) x[i]={d(g),d(g)};
    fastfft::Stockham8Plan::useRadix16=true; fastfft::Stockham8Plan sp(N); fastfft::FourStepPlan fs; fs.init(N); fastfft::Plan pl(N);
    int n=int(N);
    fftwf_plan fp=fftwf_plan_dft_1d(n,(fftwf_complex*)x,(fftwf_complex*)z,FFTW_FORWARD,FFTW_MEASURE);
    for(size_t i=0;i<total;i++) x[i]={d(g),d(g)};
    double err[3]={0,0,0}, m=0;
    for(size_t b=0;b<B;b++) fftwf_execute_dft(fp,(fftwf_complex*)(x+b*N),(fftwf_complex*)(z+b*N));
    for(size_t i=0;i<total;i++) m=std::max(m,(double)std::abs(z[i]));
    for(int v=0;v<3;v++){ for(size_t b=0;b<B;b++){ if(v==0) sp.forward(x+b*N,y+b*N); else if(v==1) fs.forward((float*)(x+b*N),(float*)(y+b*N)); else pl.forward(x+b*N,y+b*N);} for(size_t i=0;i<total;i++) err[v]=std::max(err[v],(double)std::abs(y[i]-z[i])/m);}
    double ts=timeIt([&]{for(size_t b=0;b<B;b++) sp.forward(x+b*N,y+b*N);})/B;
    double t4=timeIt([&]{for(size_t b=0;b<B;b++) fs.forward((float*)(x+b*N),(float*)(y+b*N));})/B;
    double tp=timeIt([&]{for(size_t b=0;b<B;b++) pl.forward(x+b*N,y+b*N);})/B;
    double tw=timeIt([&]{for(size_t b=0;b<B;b++) fftwf_execute_dft(fp,(fftwf_complex*)(x+b*N),(fftwf_complex*)(z+b*N));})/B;
    double fl=5.0*N*std::log2(double(N));
    printf("%8zu %10.0f %10.0f %10.0f %10.0f | %6.1f %6.1f %6.2f %s\n",N,ts*1e9,t4*1e9,tp*1e9,tw*1e9,fl/ts*1e-9,fl/t4*1e-9,tw/std::min(ts,t4), (err[0]>1e-5||err[1]>1e-5||err[2]>1e-5)?"ERR":"");
    fftwf_destroy_plan(fp); free(x); free(y); free(z);
  }
}
