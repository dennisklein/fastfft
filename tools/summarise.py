# Stream-mode time of every bm_fft case relative to SimdFFT, one markdown table per (target, type).
#   python3 tools/summarise.py results/<host>/bm-*.txt > summary.md
# SPDX-FileCopyrightText: 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later WITH LGPL-3.0-linking-exception
import collections
import pathlib
import re
import sys

rows = collections.defaultdict(dict)
for path in sys.argv[1:]:
    target = pathlib.Path(path).stem.removeprefix("bm-")
    for line in open(path, encoding="utf-8"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        match = re.match(r"│ (float|double) N=(\d+) stream: (.+?)\s+│.*│\s*([\d.]+)\s*│\s*$", line)
        if match:
            type_, n, case, ratio = match.groups()
            rows[(target, type_, int(n))][case] = float(ratio)

for target in dict.fromkeys(key[0] for key in rows):
    for type_ in ("float", "double"):
        sizes = sorted(n for (t, ty, n) in rows if t == target and ty == type_)
        if not sizes:
            continue
        others = list(dict.fromkeys(c for n in sizes for c in rows[(target, type_, n)] if not c.startswith(("Stockham", "SimdFFT"))))
        print(f"\n### {target} {type_} (stream, time relative to SimdFFT; < 1 is faster; * marks the default plan)\n")
        print("| N | Stockham default | best Stockham plan | " + " | ".join(others) + " |")
        print("|---" * (3 + len(others)) + "|")
        for n in sizes:
            cases = rows[(target, type_, n)]
            plans = {c: r for c, r in cases.items() if c.startswith("Stockham")}
            if not plans:
                continue  # sizes without a StockhamFFT case (non-powers of two)
            default = next((r for c, r in plans.items() if c.endswith("default")), float("nan"))
            best = min(plans, key=plans.get)
            label = best.removeprefix("Stockham ").replace(" default", "*")
            cells = " | ".join(f"{cases[c]:.2f}" if c in cases else "" for c in others)
            print(f"| 2^{n.bit_length() - 1} | {default:.2f} | {plans[best]:.2f} {label} | {cells} |")
