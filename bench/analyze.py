#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later WITH LGPL-3.0-linking-exception
"""Summarise fftbench raw samples (python3 standard library only).

usage: analyze.py RESULT_DIR   (reads every RESULT_DIR/*raw*.csv; writes summary.md, summary.csv, speedup.csv there)

Each raw file is one process run; samples of all runs are pooled.

For every (label, impl, precision, n): the median time per transform with a 95% percentile-bootstrap confidence
interval, and the relative median absolute deviation (MAD/median) as a spread measure. For every baseline B and size:
the speedup of fastfft over B as the median of the per-round ratios t_B / t_fastfft (samples from the same round were
taken back to back in random order, so drifts cancel), again with a 95% bootstrap CI. Only binaries that measured both
implementations in the same run are paired.
"""
import csv, glob, os, random, statistics, sys
from collections import defaultdict

B = 4000  # bootstrap resamples
rng = random.Random(1)


def boot_ci(xs, stat=statistics.median, level=0.95):
    if len(xs) < 2:
        return xs[0], xs[0]
    reps = sorted(stat(rng.choices(xs, k=len(xs))) for _ in range(B))
    lo = reps[int((1 - level) / 2 * B)]
    hi = reps[int((1 + level) / 2 * B) - 1]
    return lo, hi


def main(d):
    t = defaultdict(dict)  # (label, impl, prec, n) -> (file, round) -> ns
    for f in sorted(glob.glob(os.path.join(d, "*raw*.csv"))):
        if f.endswith("_accuracy.csv"):
            continue
        for r in csv.DictReader(open(f)):
            t[(r["label"], r["impl"], r["precision"], int(r["n"]))][(f, int(r["round"]))] = float(r["ns"])
    if not t:
        sys.exit(f"no raw csv in {d}")

    rows = []
    for (label, impl, prec, n), s in sorted(t.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1], kv[0][3])):
        xs = list(s.values())
        med = statistics.median(xs)
        mad = statistics.median(abs(x - med) for x in xs)
        lo, hi = boot_ci(xs)
        rows.append(dict(label=label, impl=impl, precision=prec, n=n, samples=len(xs), median_ns=med, ci_lo=lo, ci_hi=hi,
                         rel_mad=mad / med, gflops=5 * n * (n.bit_length() - 1) / med))

    speed = []
    for (label, impl, prec, n), s in t.items():
        if impl == "fastfft":
            continue
        f = t.get((label, "fastfft", prec, n))
        if not f:
            continue
        ratios = [s[r] / f[r] for r in s if r in f]
        lo, hi = boot_ci(ratios)
        speed.append(dict(label=label, baseline=impl, precision=prec, n=n, pairs=len(ratios),
                          speedup=statistics.median(ratios), ci_lo=lo, ci_hi=hi))
    speed.sort(key=lambda r: (r["precision"], r["label"], r["baseline"], r["n"]))

    with open(os.path.join(d, "summary.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(d, "speedup.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(speed[0]) if speed else ["none"])
        w.writeheader()
        w.writerows(speed)

    out = []
    for prec in ("float", "double"):
        cols = sorted({(r["label"], r["impl"]) for r in rows if r["precision"] == prec},
                      key=lambda c: (c[1] != "fastfft", c[0], c[1]))
        if not cols:
            continue
        idx = {(r["label"], r["impl"], r["n"]): r for r in rows if r["precision"] == prec}
        ns = sorted({r["n"] for r in rows if r["precision"] == prec})
        out.append(f"\n### complex<{prec}>: median ns per transform [95% CI]\n")
        out.append("| N | " + " | ".join(f"{i} ({l})" for l, i in cols) + " |")
        out.append("|---:|" + "---:|" * len(cols))
        for n in ns:
            cells = []
            for l, i in cols:
                r = idx.get((l, i, n))
                cells.append(f"{r['median_ns']:.4g} [{r['ci_lo']:.4g}, {r['ci_hi']:.4g}]" if r else "")
            out.append(f"| {n} | " + " | ".join(cells) + " |")
        sp = [s for s in speed if s["precision"] == prec]
        bases = sorted({(s["label"], s["baseline"]) for s in sp})
        if bases:
            sidx = {(s["label"], s["baseline"], s["n"]): s for s in sp}
            out.append(f"\n### complex<{prec}>: speedup of fastfft, median of paired ratios [95% CI]\n")
            out.append("| N | " + " | ".join(f"vs {b} ({l})" for l, b in bases) + " |")
            out.append("|---:|" + "---:|" * len(bases))
            for n in ns:
                cells = []
                for l, b in bases:
                    s = sidx.get((l, b, n))
                    cells.append(f"{s['speedup']:.2f} [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]" if s else "")
                out.append(f"| {n} | " + " | ".join(cells) + " |")
    worst = max(r["rel_mad"] for r in rows)
    out.append(f"\nSamples per cell: {min(r['samples'] for r in rows)}..{max(r['samples'] for r in rows)}; "
               f"largest relative MAD: {worst:.1%}.")
    text = "\n".join(out).lstrip() + "\n"
    open(os.path.join(d, "summary.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
