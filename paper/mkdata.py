#!/usr/bin/env python3
"""Generate the paper's numbers.tex, tables.tex and data/*.dat from a results directory.

usage: mkdata.py RESULTS_DIR [GR4_RESULTS_DIR]     (run bench/analyze.py on RESULTS_DIR first)
"""
import csv, os, re, sys

here = os.path.dirname(os.path.abspath(__file__))
res = sys.argv[1]
gr4 = sys.argv[2] if len(sys.argv) > 2 else None
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
\\caption{{Complex {prec}: median ns per transform [95\\,\\% CI] and speedup of fastfft (median of paired ratios [95\\,\\% CI]).}}\\label{{{tab}}}
\\begin{{tabular}}{{rlllll}}\\toprule
$N$ & fastfft & gr4 SimdFFT & FFTW 3.3.10 & vs.\\ SimdFFT & vs.\\ FFTW\\\\\\midrule""")
    for n in NS:
        tables.append(f"$2^{{{n.bit_length() - 1}}}$ & {cell(NATIVE, 'fastfft', prec, n)} & {cell(NATIVE, 'gr4-SimdFFT', prec, n)} & "
                      f"{cell(NATIVE, 'fftw', prec, n)} & {scell(NATIVE, 'gr4-SimdFFT', prec, n)} & {scell(NATIVE, 'fftw', prec, n)}\\\\")
    tables.append("\\bottomrule\\end{tabular}\\end{table}\n")

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

tables.append(f"""\\begin{{table}}[h]\\centering
\\caption{{Accuracy: largest $\\max_k|X_k-\\hat X_k|/\\max_k|\\hat X_k|$ over all $N$, against a long-double reference.}}\\label{{tab:acc}}
\\begin{{tabular}}{{lccc}}\\toprule & fastfft & SimdFFT & FFTW\\\\\\midrule
float & {sci(maxerr('fastfft', 'float'))} & {sci(maxerr('gr4-SimdFFT', 'float'))} & {sci(maxerr('fftw', 'float'))}\\\\
double & {sci(maxerr('fastfft', 'double'))} & {sci(maxerr('gr4-SimdFFT', 'double'))} & {sci(maxerr('fftw', 'double'))}\\\\
\\bottomrule\\end{{tabular}}\\end{{table}}
""")

num = {}
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

with open(os.path.join(here, "numbers.tex"), "w") as f:
    for k, v in sorted(num.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
with open(os.path.join(here, "tables.tex"), "w") as f:
    f.write("\n".join(tables))
print("\n".join(f"{k} = {v}" for k, v in sorted(num.items())))
