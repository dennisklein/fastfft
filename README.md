# StockhamFFT for GNU Radio 4

Work towards a gnuradio4 pull request: a measured, portable `std::simd` FFT kernel family next to `SimdFFT` (`StockhamFFT`), its switch table, and the benchmark that tunes it.
The background experiment and paper are on the `main` branch.

- `doc/design.md`: design under discussion
- `gr4/patches/`: the patch series for gnuradio4 at `gr4/REVISION`; `gr4/apply.sh [DIR]` clones and applies it

Code is LGPL-3.0-or-later WITH LGPL-3.0-linking-exception, documentation CC-BY-4.0 (see `REUSE.toml`).
Copyright 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH; author Dennis Klein <d.klein@gsi.de>.
