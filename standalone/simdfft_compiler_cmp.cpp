#include <gnuradio-4.0/algorithm/fourier/SimdFFT.hpp>
#include <chrono>
#include <complex>
#include <cstdio>
#include <random>
template<class F> double timeIt(F&& f){ using clk=std::chrono::steady_clock; int reps=1;
  for(;;){auto t0=clk::now(); for(int i=0;i<reps;i++)f(); double d=std::chrono::duration<double>(clk::now()-t0).count(); if(d>1e-3)break; reps*=2;}
  double best=1e9; for(int k=0;k<31;k++){auto t0=clk::now(); for(int i=0;i<reps;i++)f(); best=std::min(best,std::chrono::duration<double>(clk::now()-t0).count()/reps);} return best;}
int main(){ using namespace gr::algorithm; std::mt19937 g(1); std::uniform_real_distribution<float> d(-1,1);
  for(size_t N : {64UL,1024UL,4096UL,65536UL,1048576UL}){ std::vector<float, gr::allocator::Aligned<float>> x(2*N), y(2*N); for(auto& v:x) v=d(g);
    SimdFFT<float, Transform::Complex> f; f.resize(N);
    double t=timeIt([&]{ f.template transform<Direction::Forward, Order::Ordered>(std::span<const float>(x), std::span<float>(y)); });
    printf("%8zu %10.0f\n",N,t*1e9);} }
// link stubs for the two out-of-line gr::exception members (only reached on error paths)
gr::exception::exception(std::string_view, std::source_location) noexcept {}
const char* gr::exception::what() const noexcept { return "gr::exception"; }
