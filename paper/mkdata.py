#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later WITH LGPL-3.0-linking-exception
"""Generate the paper's numbers.tex, tables.tex and data/*.dat from a results directory.

usage: mkdata.py RESULTS_DIR [GR4_RESULTS_DIR] [--host2 RESULTS_DIR2]   (run bench/analyze.py on the results first)

--host2 adds a second host (macros \\Two..., appendix tables tab:two-float/-double); it needs the gcc17 labels.
"""
import csv, os, re, sys

here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
host2 = None
if "--host2" in argv:
    i = argv.index("--host2")
    host2 = argv[i + 1]
    del argv[i:i + 2]
res = argv[0]
gr4 = argv[1] if len(argv) > 1 else None
os.makedirs(os.path.join(here, "data"), exist_ok=True)

summ = list(csv.DictReader(open(os.path.join(res, "summary.csv"))))
speed = list(csv.DictReader(open(os.path.join(res, "speedup.csv"))))
acc = list(csv.DictReader(open(os.path.join(res, "check_accuracy.csv"))))
host = open(os.path.join(res, "host.txt")).read()
NATIVE = "w64" if any(r["label"] == "w64" for r in summ) else sorted({r["label"] for r in summ if r["label"].startswith("w")})[-1]

T = {(r["label"], r["impl"], r["precision"], int(r["n"])): r for r in summ}
S = {(r["label"], r["baseline"], r["precision"], int(r["n"])): r for r in speed}
NS = sorted({int(r["n"]) for r in summ})


def sp(base, prec, label=NATIVE):
    return [(n, S[(label, base, prec, n)]) for n in NS if (label, base, prec, n) in S]


def rng_(xs):
    v = [float(s["speedup"]) for _, s in xs]
    return f"{min(v):.1f}--{max(v):.1f}$\\times$" if v else "n/a"


def rng2(xs):
    v = [float(s["speedup"]) for _, s in xs]
    return f"{min(v):.2f}--{max(v):.2f}$\\times$" if v else "n/a"


# --- figure data
for prec in ("float", "double"):
    for base, tag in (("gr4-SimdFFT", "gr4"), ("fftw", "fftw")):
        with open(os.path.join(here, "data", f"{prec}-{tag}.dat"), "w") as f:
            f.write("n s em ep\n")
            for n, s in sp(base, prec):
                v, lo, hi = float(s["speedup"]), float(s["ci_lo"]), float(s["ci_hi"])
                f.write(f"{n} {v:.4f} {v - lo:.4f} {hi - v:.4f}\n")


def ns(x):
    x = float(x)
    if x >= 1e6:
        return f"{x / 1e6:.3f}\\,M"
    if x >= 1e4:
        return f"{x / 1e3:.1f}\\,k"
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


def cell(label, impl, prec, n):
    r = T.get((label, impl, prec, n))
    if not r:
        return "--"
    return f"{ns(r['median_ns'])} {{\\scriptsize[{ns(r['ci_lo'])}, {ns(r['ci_hi'])}]}}"


XT = "xtick={64,256,1024,4096,16384,65536,262144,1048576},xticklabels={$2^6$,$2^8$,$2^{10}$,$2^{12}$,$2^{14}$,$2^{16}$,$2^{18}$,$2^{20}$}"


def fig(key, caption, series, src=None):
    """Plot of median time / (N log2 N) with the 95% CI as error bars, one line per (legend, label, impl, prec)."""
    src = T if src is None else src
    plots = []
    for i, (leg, label, impl, prec) in enumerate(series):
        rows = [(n, src[(label, impl, prec, n)]) for n in NS if (label, impl, prec, n) in src]
        if not rows:
            continue
        fn = f"data/plot-{key}-{i}.dat"
        with open(os.path.join(here, fn), "w") as f:
            f.write("n y em ep\n")
            for n, r in rows:
                d = n * (n.bit_length() - 1)
                y, lo, hi = float(r["median_ns"]) / d, float(r["ci_lo"]) / d, float(r["ci_hi"]) / d
                f.write(f"{n} {y:.5f} {y - lo:.5f} {hi - y:.5f}\n")
        plots.append(f"\\addplot+[mark size=1.3pt,error bars/.cd,y dir=both,y explicit] table[x=n,y=y,y error plus=ep,y error minus=em]{{{fn}}};"
                     f"\\addlegendentry{{{leg}}}")
    return (f"\\begin{{figure}}[h]\\centering\\begin{{tikzpicture}}\\begin{{axis}}[width=0.8\\textwidth,height=62mm,"
            f"xmode=log,log basis x=2,{XT},xlabel={{transform size $N$}},ylabel={{ns / $(N\\log_2 N)$}},grid=major,"
            f"grid style={{gray!25}},legend style={{font=\\footnotesize,at={{(0.5,1.03)}},anchor=south,draw=none}},legend columns=3,"
            f"tick label style={{font=\\footnotesize}}]\n" + "\n".join(plots) +
            f"\n\\end{{axis}}\\end{{tikzpicture}}\\caption{{{caption} Time per transform divided by $N\\log_2 N$ "
            f"(lower is faster), error bars: 95\\,\\% CI of the median.}}\\label{{fig:{key}}}\\end{{figure}}\n")


def scell(label, base, prec, n):
    s = S.get((label, base, prec, n))
    if not s:
        return "--"
    return f"{float(s['speedup']):.2f} {{\\scriptsize[{float(s['ci_lo']):.2f}, {float(s['ci_hi']):.2f}]}}"


def maxerr(impl, prec):
    v = [float(r["rel_err"]) for r in acc if r["impl"] == impl and r["precision"] == prec]
    return max(v) if v else float("nan")


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"\\num{{{m}e{int(e)}}}"


tables = []
for prec, tab in (("float", "tab:float"), ("double", "tab:double")):
    tables.append(f"""\\begin{{table}}[h]\\centering
\\caption{{Complex {prec}, GCC 14.2: median ns per transform [95\\,\\% CI] and speedup of fastfft (median of paired ratios [95\\,\\% CI]).}}\\label{{{tab}}}
\\begin{{tabular}}{{rlllll}}\\toprule
$N$ & fastfft & gr4 SimdFFT & FFTW 3.3.10 & vs.\\ SimdFFT & vs.\\ FFTW\\\\\\midrule""")
    for n in NS:
        tables.append(f"$2^{{{n.bit_length() - 1}}}$ & {cell(NATIVE, 'fastfft', prec, n)} & {cell(NATIVE, 'gr4-SimdFFT', prec, n)} & "
                      f"{cell(NATIVE, 'fftw', prec, n)} & {scell(NATIVE, 'gr4-SimdFFT', prec, n)} & {scell(NATIVE, 'fftw', prec, n)}\\\\")
    tables.append("\\bottomrule\\end{tabular}\\end{table}\n")
    tables.append(fig(prec, f"Complex {prec}, GCC 14.2 (Table~\\ref{{{tab}}}).", [("fastfft", NATIVE, "fastfft", prec), ("SimdFFT", NATIVE, "gr4-SimdFFT", prec), ("FFTW", NATIVE, "fftw", prec)]))

widths = sorted({r["label"] for r in summ if re.fullmatch(r"w\d+", r["label"])}, key=lambda l: int(l[1:]))
if len(widths) > 1:
    tables.append(f"""\\begin{{table}}[h]\\centering
\\caption{{Vector width, complex float: fastfft compiled for {', '.join(str(int(w[1:]) * 8) for w in widths)}-bit vectors
(\\texttt{{FASTFFT\\_SIMD\\_BYTES}}), median ns per transform [95\\,\\% CI]; SimdFFT for reference.}}\\label{{tab:width}}
\\begin{{tabular}}{{r{'l' * (len(widths) + 1)}}}\\toprule
$N$ & {' & '.join(f'{int(w[1:]) * 8} bit' for w in widths)} & SimdFFT\\\\\\midrule""")
    for n in NS:
        tables.append(f"$2^{{{n.bit_length() - 1}}}$ & " + " & ".join(cell(w, "fastfft", "float", n) for w in widths)
                      + f" & {cell(NATIVE, 'gr4-SimdFFT', 'float', n)}\\\\")
    tables.append("\\bottomrule\\end{tabular}\\end{table}\n")
    tables.append(fig("width", "Vector width, complex float (Table~\\ref{tab:width}).", [(f"{int(w[1:]) * 8} bit", w, "fastfft", "float") for w in widths] + [("SimdFFT", NATIVE, "gr4-SimdFFT", "float")]))

if any(r["label"] == "libcxx" for r in summ):
    tables.append("""\\begin{table}[h]\\centering
\\caption{Standard library: complex float built with GCC/libstdc++ (native simd) and with clang/libc++ (vir-simd array
emulation), median ns per transform [95\\,\\% CI].}\\label{tab:libcxx}
\\begin{tabular}{rllll}\\toprule
$N$ & fastfft, libstdc++ & fastfft, libc++ & SimdFFT, libstdc++ & SimdFFT, libc++\\\\\\midrule""")
    for n in NS:
        tables.append(f"$2^{{{n.bit_length() - 1}}}$ & {cell(NATIVE, 'fastfft', 'float', n)} & {cell('libcxx', 'fastfft', 'float', n)} & "
                      f"{cell(NATIVE, 'gr4-SimdFFT', 'float', n)} & {cell('libcxx', 'gr4-SimdFFT', 'float', n)}\\\\")
    tables.append("\\bottomrule\\end{tabular}\\end{table}\n")
    tables.append(fig("libcxx", "Standard library, complex float (Table~\\ref{tab:libcxx}).", [("fastfft, libstdc++", NATIVE, "fastfft", "float"), ("fastfft, libc++", "libcxx", "fastfft", "float"), ("SimdFFT, libstdc++", NATIVE, "gr4-SimdFFT", "float"), ("SimdFFT, libc++", "libcxx", "gr4-SimdFFT", "float")]))

num_tr = {}
if any(r["label"] == "gcc17" for r in summ):
    def med(l, i, p, n):
        r = T.get((l, i, p, n))
        return float(r["median_ns"]) if r else None

    for prec in ("float", "double"):
        tables.append(f"""\\begin{{table}}[h]\\centering
\\caption{{GCC trunk (17.0) with \\texttt{{-std=c++26}}, complex {prec}: median ns per transform [95\\,\\% CI].
fastfft on \\texttt{{<experimental/simd>}} (GCC 14 and trunk), on C++26 \\texttt{{std::simd}}, and fastfft::cx on
\\texttt{{std::simd::vec<std::complex<{prec}>{{}}>}} (plain, and with zip/broadcast via \\texttt{{bit\\_cast}} to a real
vector); SimdFFT built with trunk for reference.}}\\label{{tab:gcc17-{prec}}}
\\begin{{tabular}}{{rrrrrrr}}\\toprule
$N$ & exp., GCC 14 & exp., trunk & C++26 & cx & cx, bit\\_cast & SimdFFT\\\\\\midrule""")
        def scell(*a):  # stacked: median above its CI, so seven columns fit the page
            m, ci = cell(*a).split(" ", 1)
            return f"\\begin{{tabular}}[t]{{@{{}}r@{{}}}}{m}\\\\{ci}\\end{{tabular}}"
        for n in NS:
            tables.append(f"$2^{{{n.bit_length() - 1}}}$ & {scell(NATIVE, 'fastfft', prec, n)} & {scell('gcc17-exp', 'fastfft', prec, n)} & "
                          f"{scell('gcc17', 'fastfft', prec, n)} & {scell('gcc17', 'fastfft-cx', prec, n)} & "
                          f"{scell('gcc17-cxbc', 'fastfft-cx', prec, n)} & {scell('gcc17', 'gr4-SimdFFT', prec, n)}\\\\")
        tables.append("\\bottomrule\\end{tabular}\\end{table}\n")
        tables.append(fig(f"gcc17-{prec}", f"GCC trunk, complex {prec} (Table~\\ref{{tab:gcc17-{prec}}}).", [("exp., GCC 14", NATIVE, "fastfft", prec), ("exp., trunk", "gcc17-exp", "fastfft", prec), ("C++26", "gcc17", "fastfft", prec), ("cx", "gcc17", "fastfft-cx", prec), ("cx, bit\\_cast", "gcc17-cxbc", "fastfft-cx", prec), ("SimdFFT", "gcc17", "gr4-SimdFFT", prec)]))

    def ratios(num_, den_, prec, lo=0, hi=1 << 30):
        v = [med(*num_, prec, n) / med(*den_, prec, n) for n in NS if lo <= n <= hi and med(*num_, prec, n) and med(*den_, prec, n)]
        return v

    def rr(v, digits=2):
        return f"{min(v):.{digits}f}--{max(v):.{digits}f}"

    for prec, P in (("float", "Float"), ("double", "Double")):
        # >1 means slower than the reference in the second argument
        num_tr[f"{P}CxxSmall"] = rr(ratios(("gcc17", "fastfft"), ("gcc17-exp", "fastfft"), prec, hi=512))
        num_tr[f"{P}CxxLarge"] = rr(ratios(("gcc17", "fastfft"), ("gcc17-exp", "fastfft"), prec, lo=1024))
        num_tr[f"{P}CxSmall"] = rr(ratios(("gcc17", "fastfft-cx"), ("gcc17", "fastfft"), prec, hi=1 << 15))
        num_tr[f"{P}CxLarge"] = rr(ratios(("gcc17", "fastfft-cx"), ("gcc17", "fastfft"), prec, lo=1 << 16))
        num_tr[f"{P}CxbcSmall"] = rr(ratios(("gcc17-cxbc", "fastfft-cx"), ("gcc17-cxbc", "fastfft"), prec, hi=1 << 15))
        num_tr[f"{P}CxbcLarge"] = rr(ratios(("gcc17-cxbc", "fastfft-cx"), ("gcc17-cxbc", "fastfft"), prec, lo=1 << 16))
        num_tr[f"{P}TrunkVsGr"] = rr(ratios(("gcc17", "gr4-SimdFFT"), ("gcc17", "fastfft"), prec), 1)
    num_tr["CxErr"] = sci(max(float(r["rel_err"]) for r in csv.DictReader(open(os.path.join(res, "check-gcc17_accuracy.csv"))) if r["impl"] == "fastfft-cx" and r["precision"] == "float"))
    num_tr["CxErrD"] = sci(max(float(r["rel_err"]) for r in csv.DictReader(open(os.path.join(res, "check-gcc17_accuracy.csv"))) if r["impl"] == "fastfft-cx" and r["precision"] == "double"))
    num_tr["TrunkVersion"] = re.search(r"gcc trunk: (.*)", host).group(1).split("(experimental)")[0].replace("g++ (GCC) ", "").strip() if "gcc trunk:" in host else "?"
    num_tr["TrunkCommit"] = (re.search(r"gcc trunk: .*\) (\w+)", host) or re.search(r"(x)", "x")).group(1)[:10]

tables.append(f"""\\begin{{table}}[h]\\centering
\\caption{{Accuracy: largest $\\max_k|X_k-\\hat X_k|/\\max_k|\\hat X_k|$ over all $N$, against a long-double reference.}}\\label{{tab:acc}}
\\begin{{tabular}}{{lccc}}\\toprule & fastfft & SimdFFT & FFTW\\\\\\midrule
float & {sci(maxerr('fastfft', 'float'))} & {sci(maxerr('gr4-SimdFFT', 'float'))} & {sci(maxerr('fftw', 'float'))}\\\\
double & {sci(maxerr('fastfft', 'double'))} & {sci(maxerr('gr4-SimdFFT', 'double'))} & {sci(maxerr('fftw', 'double'))}\\\\
\\bottomrule\\end{{tabular}}\\end{{table}}
""")

num = dict(num_tr)
num["FloatVsGrRange"] = rng_(sp("gr4-SimdFFT", "float"))
num["DoubleVsGrRange"] = rng_(sp("gr4-SimdFFT", "double"))
num["FloatVsFftwRange"] = rng2(sp("fftw", "float"))
num["DoubleVsFftwRange"] = rng2(sp("fftw", "double"))
num["FloatErr"], num["DoubleErr"] = sci(maxerr("fastfft", "float")), sci(maxerr("fastfft", "double"))
num["FloatErrGr"], num["DoubleErrGr"] = sci(maxerr("gr4-SimdFFT", "float")), sci(maxerr("gr4-SimdFFT", "double"))
num["WorstMAD"] = f"{max(float(r['rel_mad']) for r in summ) * 100:.1f}\\,\\%"
num["Samples"] = str(min(int(r["samples"]) for r in summ))
if len(widths) > 1:
    w0 = widths[0]
    r16 = [float(T[(NATIVE, "gr4-SimdFFT", "float", n)]["median_ns"]) / float(T[(w0, "fastfft", "float", n)]["median_ns"]) for n in NS
           if (w0, "fastfft", "float", n) in T and (NATIVE, "gr4-SimdFFT", "float", n) in T]
    g = [float(T[(w0, "fastfft", "float", n)]["median_ns"]) / float(T[(NATIVE, "fastfft", "float", n)]["median_ns"]) for n in NS
         if (w0, "fastfft", "float", n) in T]
    num["WidthSixteenVsGr"] = f"{min(r16):.1f}--{max(r16):.1f}$\\times$"
    num["WidthGain"] = f"{min(g):.1f}--{max(g):.1f}"
if any(r["label"] == "libcxx" for r in summ):
    f = [float(T[("libcxx", "fastfft", "float", n)]["median_ns"]) / float(T[(NATIVE, "fastfft", "float", n)]["median_ns"]) for n in NS if ("libcxx", "fastfft", "float", n) in T]
    g = [float(T[("libcxx", "gr4-SimdFFT", "float", n)]["median_ns"]) / float(T[(NATIVE, "gr4-SimdFFT", "float", n)]["median_ns"]) for n in NS if ("libcxx", "gr4-SimdFFT", "float", n) in T]
    num["FallbackSlowdown"] = f"{min(f):.1f}--{max(f):.1f}$\\times$"
    num["FallbackSlowdownGr"] = f"{min(g):.1f}--{max(g):.1f}$\\times$"


def hostfield(key):
    m = re.search(rf"^\s*{key}:\s*(.+)$", host, re.M)
    return m.group(1).strip() if m else "?"


num["HostCPU"] = hostfield("Model name").replace("(R)", "").replace("(TM)", "").replace("CPU ", "").replace("  ", " ")
num["HostKernel"] = hostfield("uname").split()[2]
num["HostCompiler"] = re.sub(r"\s*\(.*?\)", "", hostfield("compiler"))
num["HostCores"] = f"{hostfield('CPU\\(s\\)')} vCPU"

if gr4:
    rows = []
    for line in open(os.path.join(gr4, "summary.md"), encoding="utf-8"):
        m = re.match(r"\| (.+?) \| (.+?) \| (.+?) \| ([\d.]+) \[([\d.]+), ([\d.]+)\] \|", line)
        if m:
            rows.append(m.groups())
    tables.append("""\\begin{table}[h]\\centering
\\caption{gnuradio4 blocks before and after the patch, gnuradio4's benchmark facility: median call time over runs,
speedup as median of paired per-run ratios [95\\,\\% CI].}\\label{tab:gr4}
\\begin{tabular}{llll}\\toprule benchmark row & unpatched & patched & speedup\\\\\\midrule""")
    for name, b, p, s, lo, hi in rows:
        tables.append(f"\\texttt{{{name.replace('<', '$<$').replace('>', '$>$').replace('_', '\\_')}}} & {b.replace('µs', '\\,\\textmu s').replace(' ns', '\\,ns')} & {p.replace('µs', '\\,\\textmu s').replace(' ns', '\\,ns')} & {s} {{\\scriptsize[{lo}, {hi}]}}\\\\")
    tables.append("\\bottomrule\\end{tabular}\\end{table}\n")
    with open(os.path.join(here, "data", "plot-gr4.dat"), "w") as f:
        f.write("i s em ep\n")
        for k, (name, b, p, sv, lo, hi) in enumerate(rows):
            f.write(f"{k} {sv} {float(sv) - float(lo):.3f} {float(hi) - float(sv):.3f}\n")
    labels = ",".join("{\\texttt{" + re.sub(r"\s+", " ", n).replace("<", "$<$").replace(">", "$>$").replace("_", "\\_") + "}}" for n, *_ in rows)
    tables.append("\\begin{figure}[h]\\centering\\begin{tikzpicture}\\begin{axis}[xbar,width=0.5\\textwidth,"
                  f"height={6 + 5 * len(rows)}mm,y dir=reverse,ytick={{0,...,{len(rows) - 1}}},yticklabels={{{labels}}},"
                  "xlabel={speedup of the patched build},xmin=0,grid=major,grid style={gray!25},tick label style={font=\\scriptsize},"
                  "bar width=3pt,enlarge y limits=0.03,extra x ticks={1},extra x tick style={grid style={black}}]\n"
                  "\\addplot+[error bars/.cd,x dir=both,x explicit] table[x=s,y=i,x error plus=ep,x error minus=em]{data/plot-gr4.dat};\n"
                  "\\end{axis}\\end{tikzpicture}\\caption{gnuradio4 blocks, speedup of the patched build (Table~\\ref{tab:gr4}), "
                  "error bars: 95\\,\\% CI of the median of paired ratios.}\\label{fig:gr4}\\end{figure}\n")
    st = {"float": [], "double": []}
    for name, b, p, s, lo, hi in rows:
        m = re.match(r"FFT<(float32|float64|float|double)> stream", name)
        if m:
            st[{"float32": "float", "float64": "double"}.get(m.group(1), m.group(1))].append(float(s))
        if "complex<float> - fft" in name and "fftCT" not in name:
            num["GrSpectrumComplex"] = f"{float(s):.2f}$\\times$"
        if name.startswith("float - fft"):
            num["GrSpectrumReal"] = f"{float(s):.2f}$\\times$"
    for k, v in st.items():
        if v:
            num[f"GrStream{k.capitalize()}"] = f"{min(v):.1f}--{max(v):.1f}$\\times$"

if host2:
    summ2 = list(csv.DictReader(open(os.path.join(host2, "summary.csv"))))
    T2 = {(r["label"], r["impl"], r["precision"], int(r["n"])): r for r in summ2}
    S2 = {(r["label"], r["baseline"], r["precision"], int(r["n"])): r for r in csv.DictReader(open(os.path.join(host2, "speedup.csv")))}
    h2 = open(os.path.join(host2, "host.txt")).read()
    N2 = "w64" if any(r["label"] == "w64" for r in summ2) else sorted({r["label"] for r in summ2 if r["label"].startswith("w")})[-1]

    def stack(m, ci):
        return "\\begin{tabular}[t]{@{}r@{}}" + m + "\\\\{\\scriptsize" + ci + "}\\end{tabular}"

    def c2(label, impl, prec, n):
        r = T2.get((label, impl, prec, n))
        return stack(ns(r["median_ns"]), f"[{ns(r['ci_lo'])}, {ns(r['ci_hi'])}]") if r else "--"

    def s2(label, base, prec, n):
        r = S2.get((label, base, prec, n))
        return stack(f"{float(r['speedup']):.2f}", f"[{float(r['ci_lo']):.2f}, {float(r['ci_hi']):.2f}]") if r else "--"

    def m2(l, i, p, n):
        r = T2.get((l, i, p, n))
        return float(r["median_ns"]) if r else None

    def r2(num_, den_, prec, lo=0, hi=1 << 30, digits=2):
        v = [m2(*num_, prec, n) / m2(*den_, prec, n) for n in NS if lo <= n <= hi and m2(*num_, prec, n) and m2(*den_, prec, n)]
        return f"{min(v):.{digits}f}--{max(v):.{digits}f}"

    def h2field(key):
        m = re.search(rf"^\s*{key}:\s*(.+)$", h2, re.M)
        return m.group(1).strip() if m else "?"

    cpu2 = re.sub(r"\s*w/.*", "", h2field("Model name"))
    cc2 = re.sub(r"\s*\(.*?\)", "", h2field("compiler")).replace("g++ ", "GCC ").split(" 20")[0]
    for prec, P in (("float", "Float"), ("double", "Double")):
        tables.append(f"""\\begin{{table}}[h]\\centering
\\caption{{Second host, {cpu2} (one core, {cc2}), complex {prec}: median ns per transform [95\\,\\% CI], speedups of fastfft
(paired), and the GCC trunk builds (\\texttt{{-std=c++26}}): fastfft on C++26 \\texttt{{std::simd}}, on
\\texttt{{<experimental/simd>}}, and fastfft::cx.}}\\label{{tab:two-{prec}}}
\\setlength\\tabcolsep{{3.5pt}}\\begin{{tabular}}{{rrrrrrrrr}}\\toprule
$N$ & fastfft & SimdFFT & FFTW & vs.\\ SimdFFT & vs.\\ FFTW & C++26 & exp., trunk & cx\\\\\\midrule""")
        for n in NS:
            tables.append(f"$2^{{{n.bit_length() - 1}}}$ & {c2(N2, 'fastfft', prec, n)} & {c2(N2, 'gr4-SimdFFT', prec, n)} & {c2(N2, 'fftw', prec, n)} & "
                          f"{s2(N2, 'gr4-SimdFFT', prec, n)} & {s2(N2, 'fftw', prec, n)} & {c2('gcc17', 'fastfft', prec, n)} & "
                          f"{c2('gcc17-exp', 'fastfft', prec, n)} & {c2('gcc17', 'fastfft-cx', prec, n)}\\\\")
        tables.append("\\bottomrule\\end{tabular}\\end{table}\n")
        tables.append(fig(f"two-{prec}", f"Second host, complex {prec} (Table~\\ref{{tab:two-{prec}}}).", [("fastfft", N2, "fastfft", prec), ("SimdFFT", N2, "gr4-SimdFFT", prec), ("FFTW", N2, "fftw", prec), ("C++26, trunk", "gcc17", "fastfft", prec), ("exp., trunk", "gcc17-exp", "fastfft", prec), ("cx, trunk", "gcc17", "fastfft-cx", prec)], T2))
        sv = lambda base: [float(S2[(N2, base, prec, n)]["speedup"]) for n in NS if (N2, base, prec, n) in S2]
        num[f"Two{P}VsGr"] = f"{min(sv('gr4-SimdFFT')):.1f}--{max(sv('gr4-SimdFFT')):.1f}$\\times$"
        num[f"Two{P}VsFftw"] = f"{min(sv('fftw')):.2f}--{max(sv('fftw')):.2f}$\\times$"
        num[f"Two{P}Cxx"] = r2(("gcc17", "fastfft"), ("gcc17-exp", "fastfft"), prec, hi=1 << 14)
        num[f"Two{P}CxSmall"] = r2(("gcc17", "fastfft-cx"), ("gcc17", "fastfft"), prec, hi=1 << 14)
        num[f"Two{P}CxLarge"] = r2(("gcc17", "fastfft-cx"), ("gcc17", "fastfft"), prec, lo=1 << 19)
    num["TwoCPU"] = cpu2
    num["TwoCompiler"] = cc2
    num["TwoMAD"] = f"{max(float(r['rel_mad']) for r in summ2) * 100:.1f}\\,\\%"

with open(os.path.join(here, "numbers.tex"), "w") as f:
    for k, v in sorted(num.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
with open(os.path.join(here, "tables.tex"), "w") as f:
    f.write("\n".join(tables))
print("\n".join(f"{k} = {v}" for k, v in sorted(num.items())))
