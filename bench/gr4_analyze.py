#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH, Dennis Klein
# SPDX-License-Identifier: LGPL-3.0-or-later WITH LGPL-3.0-linking-exception
"""Summarise bench/gr4_compare.sh output: RESULT_DIR/<bm>-<base|patch>-<rep>.txt -> RESULT_DIR/summary.md.

Each file is one run of a gnuradio4 benchmark binary; from every row the median call time is taken. Per row the
table gives the median over runs for base and patch and the speedup as the median of the per-run ratios
base/patch (runs alternate base and patch, so run r of both is a pair), with a 95% percentile-bootstrap CI.
"""
import glob, os, random, re, statistics, sys
from collections import defaultdict

UNIT = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0}
rng = random.Random(1)


def parse(path):
    rows, col = {}, None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("┌"):  # header: ┌──benchmark:──┬──min──┬─total time─┬─median─┬...
            names = [c.strip("─┌┐ ") for c in line.split("┬")]
            col = names.index("median") if "median" in names else None
            continue
        cells = [c.strip() for c in line.strip("│").split("│")]
        if col is not None and len(cells) > col and len(cells) >= 2 and cells[1] in ("PASS", "FAIL"):
            m = re.fullmatch(r"([\d.]+)\s*(ns|us|µs|ms|s)", cells[col])
            if m:
                rows[cells[0]] = float(m.group(1)) * UNIT[m.group(2)]
    return rows


def ci(xs, B=4000):
    reps = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(B))
    return reps[int(0.025 * B)], reps[int(0.975 * B) - 1]


def fmt(t):
    return f"{t * 1e6:.1f} µs" if t >= 1e-6 else f"{t * 1e9:.0f} ns"


def main(d):
    data = defaultdict(lambda: defaultdict(dict))  # bm -> name -> (variant, rep) -> seconds
    for f in glob.glob(os.path.join(d, "bm_*-*-*.txt")):
        bm, variant, rep = re.fullmatch(r"(bm_\w+)-(base|patch)-(\d+)\.txt", os.path.basename(f)).groups()
        for name, t in parse(f).items():
            data[bm][name][(variant, int(rep))] = t
    out = []
    for bm in sorted(data):
        out += [f"\n### {bm}: median time per call over runs; speedup = median of paired base/patch ratios [95% CI]\n",
                "| row | base | patch | speedup |", "|---|---:|---:|---:|"]
        for name, s in data[bm].items():
            reps = sorted({r for v, r in s if (("base", r) in s and ("patch", r) in s)})
            if not reps:
                continue
            b = [s[("base", r)] for r in reps]
            p = [s[("patch", r)] for r in reps]
            ratios = [x / y for x, y in zip(b, p)]
            lo, hi = ci(ratios)
            out.append(f"| {name} | {fmt(statistics.median(b))} | {fmt(statistics.median(p))} | "
                       f"{statistics.median(ratios):.2f} [{lo:.2f}, {hi:.2f}] |")
    text = "\n".join(out).lstrip() + "\n"
    open(os.path.join(d, "summary.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
