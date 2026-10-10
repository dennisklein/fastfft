### bm_fft_block: median time per call over runs; speedup = median of paired base/patch ratios [95% CI]

| row | base | patch | speedup |
|---|---:|---:|---:|
| std::complex<float> - fft | 70.5 µs | 68.9 µs | 1.06 [0.98, 1.07] |
| std::complex<float> - fftCT | 79.7 µs | 80.6 µs | 1.00 [0.95, 1.01] |
| std::complex<double> - fft | 93.0 µs | 60.2 µs | 1.55 [1.45, 1.59] |
| std::complex<double> - fftCT | 98.0 µs | 97.9 µs | 0.99 [0.91, 1.21] |
| float - fft | 23.3 µs | 22.7 µs | 1.00 [0.97, 1.07] |
| double - fft | 27.4 µs | 27.7 µs | 1.01 [0.81, 1.10] |

### bm_fft_stream: median time per call over runs; speedup = median of paired base/patch ratios [95% CI]

| row | base | patch | speedup |
|---|---:|---:|---:|
| FFT<float32> stream N=     64 | 138.3 µs | 53.5 µs | 2.55 [2.53, 4.34] |
| FFT<float32> stream N=    256 | 135.3 µs | 56.6 µs | 2.39 [2.33, 4.00] |
| FFT<float32> stream N=   1024 | 150.4 µs | 59.0 µs | 2.52 [2.49, 2.57] |
| FFT<float32> stream N=   4096 | 210.3 µs | 97.6 µs | 2.14 [2.08, 2.18] |
| FFT<float32> stream N=  16384 | 256.6 µs | 103.9 µs | 2.48 [2.37, 2.60] |
| FFT<float32> stream N=  65536 | 300.1 µs | 178.0 µs | 1.70 [1.58, 1.73] |
| FFT<float32> stream N= 262144 | 1979.0 µs | 1159.0 µs | 1.76 [1.64, 1.83] |
| FFT<float32> stream N=1048576 | 9300.0 µs | 5582.0 µs | 1.66 [1.65, 1.73] |
| FFT<float64> stream N=     64 | 195.0 µs | 144.3 µs | 1.34 [1.19, 1.43] |
| FFT<float64> stream N=    256 | 228.3 µs | 161.8 µs | 1.42 [1.32, 1.47] |
| FFT<float64> stream N=   1024 | 274.7 µs | 205.5 µs | 1.34 [1.16, 1.37] |
| FFT<float64> stream N=   4096 | 367.3 µs | 276.4 µs | 1.33 [1.26, 1.41] |
| FFT<float64> stream N=  16384 | 444.5 µs | 315.7 µs | 1.43 [1.35, 1.60] |
| FFT<float64> stream N=  65536 | 616.6 µs | 490.2 µs | 1.33 [1.16, 1.55] |
| FFT<float64> stream N= 262144 | 4268.0 µs | 2326.0 µs | 1.84 [1.74, 1.87] |
| FFT<float64> stream N=1048576 | 19490.0 µs | 11470.0 µs | 1.74 [1.66, 1.76] |
