# Merge bm_fft runs of several hosts into FFTRoutes.hpp tables: per (type, vector width, N) the candidate whose slowest host is fastest.
#   python3 tools/routes.py [--no-split] [bm-*.txt ...]   default: every results/*/bm-*.txt except C++26 and run1 files
# A candidate must be measured on every host that measured that (type, width, N); the default plan wins unless another
# candidate's worst-case ratio is lower by more than MARGIN, so noise cannot churn the tables.
# SPDX-FileCopyrightText: 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later WITH LGPL-3.0-linking-exception
import collections
import pathlib
import re
import sys

MARGIN = 0.03
ROOT = pathlib.Path(__file__).resolve().parent.parent

args = [a for a in sys.argv[1:] if not a.startswith("--")]
useSplit = "--no-split" not in sys.argv
files = [pathlib.Path(a) for a in args] or sorted(p for p in ROOT.glob("results/**/bm-*.txt") if "cxx26" not in p.name and "run1" not in p.parts)

# ratio[(type, bits, N)][host][case] = median time relative to SimdFFT
ratio = collections.defaultdict(lambda: collections.defaultdict(dict))
hosts = []
for path in files:
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8"))
    lanes = dict((t, int(w)) for t, w in re.findall(r"(float|double) (\d+)", re.search(r"^simd: .*$", text, re.M).group(0)))
    cpu = re.search(r"^cpu: (.*)$", text, re.M).group(1).strip()
    hosts.append(f"{cpu} ({path.parent.name}/{path.name})")
    for type_, n, case, value in re.findall(r"^│ (float|double) N=(\d+) stream: (.+?)\s+│.*│\s*([\d.]+)\s*│\s*$", text, re.M):
        n = int(n)
        if n & (n - 1) or not (case.startswith(("Stockham", "SimdFFT")) or (useSplit and case == "Split")):
            continue
        bits = lanes[type_] * (4 if type_ == "float" else 8) * 8
        ratio[(type_, bits, n)][str(path)][case.replace(" default", "")] = (float(value), case.endswith("default"))


def choose(byHost):
    common = set.intersection(*(set(cases) for cases in byHost.values()))
    worst = {case: max(cases[case][0] for cases in byHost.values()) for case in common}
    default = next(c for c in common if any(cases[c][1] for cases in byHost.values()))
    best = min(worst, key=worst.get)
    if worst[default] - worst[best] <= MARGIN:
        best = default
    if best != "SimdFFT" and worst[best] > 1.0 - MARGIN and worst["SimdFFT"] <= worst[best]:
        best = "SimdFFT"
    return best, best == default, worst


print("// merged by tools/routes.py from:")
for host in hosts:
    print(f"//   {host}")
for type_ in ("float", "double"):
    for bits in (128, 256, 512):
        sizes = sorted(n for (t, b, n) in ratio if t == type_ and b == bits)
        if not sizes:
            continue
        label = type_.capitalize()
        entries, plans = [], []
        for n in sizes:
            best, isDefault, worst = choose(ratio[(type_, bits, n)])
            if best == "SimdFFT":
                route = ("simdFft", None)
            elif best == "Split":
                route = ("stockhamSplit", None)
            elif isDefault:
                route = ("stockham", None)
            else:
                route = ("stockham", tuple(int(r) for r in best.removeprefix("Stockham ").split("·")))
            note = worst[best]
            if entries and entries[-1][1] == route:
                entries[-1][0] = n.bit_length() - 1
                entries[-1][2].append(note)
            else:
                entries.append([n.bit_length() - 1, route, [note]])
        if entries[-1][1][1] is not None or entries[-1][1][0] != "stockham":
            entries.append([None, ("stockham", None), []])
        print(f"\n// {label}, {bits} bit")
        lines = []
        for maxLog2N, (kernel, plan), notes in entries:
            last = (maxLog2N, (kernel, plan)) == (entries[-1][0], entries[-1][1])
            bound = "std::numeric_limits<std::size_t>::digits" if last else f"{maxLog2N}UZ"
            planArg = ""
            if plan:
                name = f"kPlan{label}{bits}Bit{len(plans)}"
                plans.append(f"inline constexpr std::array {name}{{{', '.join(f'{r}UZ' for r in plan)}}};")
                planArg = f", .plan = {name}"
            lines.append(f"    FFTRoute{{.maxLog2N = {bound}, .kernel = FFTKernel::{kernel}{planArg}}}, // {f'worst host {min(notes):.2f}-{max(notes):.2f} of SimdFFT' if notes else 'beyond the measured sizes'}")
        if plans:
            print("\n".join(plans))
        print(f"inline constexpr std::array kFFTRoutes{label}{bits}Bit{{\n" + "\n".join(lines) + "\n};")
