#include "FastFFT.hpp"
#include <cstdio>
#include <random>
using cd = std::complex<double>;
using C = std::complex<float>;
void refFFT(std::vector<cd>& a) { size_t n = a.size(); if (n == 1) return;
    std::vector<cd> e(n/2), o(n/2); for (size_t i=0;i<n/2;i++){e[i]=a[2*i];o[i]=a[2*i+1];}
    refFFT(e); refFFT(o);
    for (size_t k=0;k<n/2;k++){ cd w=std::polar(1.0,-2*std::numbers::pi*double(k)/double(n))*o[k]; a[k]=e[k]+w; a[k+n/2]=e[k]-w;} }
int main(){
  std::mt19937 g(1); std::uniform_real_distribution<float> d(-1,1); double worst=0;
  for (size_t N=8; N<=(1u<<20); N*=2){
    auto* x = (C*)std::aligned_alloc(64, N*8+64); auto* y=(C*)std::aligned_alloc(64,N*8+64);
    std::vector<cd> r(N); for(size_t i=0;i<N;i++){x[i]={d(g),d(g)}; r[i]=x[i];} refFFT(r);
    double mx=0; for(auto& v:r) mx=std::max(mx,std::abs(v));
    auto err=[&](const C* o){double e=0; for(size_t i=0;i<N;i++) e=std::max(e,std::abs(cd(o[i])-r[i])); e/=mx; worst=std::max(worst,e); return e;};
    printf("N=%7zu",N);
    { fastfft::Plan p(N); p.forward(x,y); printf(" plan=%.1e",err(y)); std::vector<C> xx(x,x+N); p.forward(xx.data(),xx.data()); printf(" inplace=%.1e",err(xx.data()));
      std::vector<C> u(N+1); std::copy(x,x+N,u.begin()+1); std::vector<C> uo(N+1); p.forward(u.data()+1,uo.data()+1); printf(" unaligned=%.1e",err(uo.data()+1)); }
    if(N>=64){ for(int r16=0;r16<2;r16++){ fastfft::Stockham8Plan::useRadix16=r16; fastfft::Stockham8Plan s(N); s.forward(x,y); printf(" st8r%d=%.1e",r16?16:8,err(y)); }
               fastfft::FourStepPlan f; f.init(N); f.forward((float*)x,(float*)y); printf(" 4step=%.1e",err(y)); }
    printf("\n");
    free(x); free(y);
  }
  printf("worst=%.2e %s\n",worst, worst<1e-6?"OK":"FAIL");
}
