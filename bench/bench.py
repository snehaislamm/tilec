#!/usr/bin/env python3
"""Performance analysis for tilec-generated kernels (single-threaded CPU).

  1. Matmul: naive i-j-k vs cache-blocked tiles vs NumPy/OpenBLAS.
  2. Tile-size sweep for the blocked matmul.
  3. Fusion: unfused vs fused pipelines on ML kernels, with the number of
     loop nests and intermediate bytes written to memory.

Writes bench/results/RESULTS.md.  Run from the repo root after building.
"""
import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import tilec  # noqa: E402

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def best_time(fn, reps):
    fn()  # warm-up (page faults, i-cache)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def matmul_src(n):
    return f"kernel mm(a: f32[{n}, {n}], b: f32[{n}, {n}]) -> f32[{n}, {n}] {{ return a @ b }}\n"


def kernels_and_traffic(path, flags):
    """Count loop nests and bytes of intermediates from the optimized IR."""
    ir = subprocess.run([tilec.tilec_binary(), str(path), "--emit=ir-opt", *flags],
                        capture_output=True, text=True, check=True).stdout
    nests, inter = 0, 0
    lines = ir.splitlines()
    ret = [l for l in lines if "tile.return" in l][0].split("%")[1].split()[0]
    for l in lines:
        s = l.strip()
        if s.startswith("%") and ("tile.fused" in s or "tile.transpose" in s):
            nests += 1
            vid = s.split()[0][1:]
            shape = s.rsplit("tensor<", 1)[1].split("xf32")[0].split("x")
            if vid != ret:
                inter += int(shape[0]) * int(shape[1]) * 4
    return nests, inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = []
    cpu = platform.processor() or platform.machine()
    try:
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
               if l.startswith("model name")][0]
    except OSError:
        pass
    out.append("# tilec benchmark results\n")
    out.append(f"CPU: {cpu}, single thread. clang {' '.join(tilec.runtime.default_cflags())}. "
               f"Best of {args.reps} runs. NumPy {np.__version__} "
               f"(OPENBLAS_NUM_THREADS={os.environ['OPENBLAS_NUM_THREADS']}).\n")

    # ---- 1. matmul variants ----
    n = args.n
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    flops = 2.0 * n ** 3
    rows = []
    variants = [("naive i-j-k (`--no-tile`)", ["--no-tile"]), ("tiled (default 32x256x128)", [])]
    ref = a @ b
    for label, flags in variants:
        k = tilec.compile_source(matmul_src(n), flags=flags)["mm"]
        o = np.empty((n, n), np.float32)
        np.testing.assert_allclose(k(a, b), ref, rtol=1e-3, atol=1e-2)
        t = best_time(lambda: k(a, b, out=o), args.reps)
        rows.append((label, t, flops / t / 1e9))
    t = best_time(lambda: a @ b, args.reps)
    rows.append(("NumPy / OpenBLAS (hand-tuned asm)", t, flops / t / 1e9))
    base = rows[0][1]
    out.append(f"\n## 1. Matmul {n}x{n}x{n}\n")
    out.append("| implementation | time (ms) | GFLOP/s | speedup vs naive |")
    out.append("|---|---:|---:|---:|")
    for label, t, g in rows:
        out.append(f"| {label} | {t * 1e3:.2f} | {g:.1f} | {base / t:.1f}x |")

    # ---- 2. tile sweep ----
    sweep = [(8, 64, 64), (16, 128, 64), (32, 128, 128), (32, 256, 128), (64, 256, 64),
             (64, 512, 256), (128, 128, 128), (16, 512, 512)]
    if args.quick:
        sweep = sweep[:3]
    out.append(f"\n## 2. Tile-size sweep (matmul {n}^3)\n")
    out.append("| TM x TN x TK | C tile + B panel (KiB) | GFLOP/s |")
    out.append("|---|---:|---:|")
    best = None
    for tm, tn, tk in sweep:
        k = tilec.compile_source(matmul_src(n), flags=[f"--tile={tm},{tn},{tk}"])["mm"]
        o = np.empty((n, n), np.float32)
        t = best_time(lambda: k(a, b, out=o), args.reps)
        g = flops / t / 1e9
        kib = (tm * tn + tk * tn) * 4 / 1024
        out.append(f"| {tm} x {tn} x {tk} | {kib:.0f} | {g:.1f} |")
        if best is None or g > best[1]:
            best = ((tm, tn, tk), g)
    out.append(f"\nBest: {best[0][0]}x{best[0][1]}x{best[0][2]} at {best[1]:.1f} GFLOP/s.")

    # ---- 3. fusion ----
    cases = {
        "mlp": ([(256, 512), (512, 128), (128,)], "relu(x @ w + b)"),
        "gelu": ([(512, 512)], "tanh-GELU, 9 elementwise ops"),
        "softmax": ([(128, 1024)], "row softmax"),
        "layernorm": ([(256, 768), (768,), (768,)], "LayerNorm"),
        "attention": ([(128, 64)] * 3, "softmax(QK^T/8)V"),
    }
    out.append("\n## 3. Kernel fusion\n")
    out.append("| kernel | unfused: loop nests / intermediates | fused: loop nests / intermediates "
               "| unfused (us) | fused (us) | speedup |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for name, (shapes, desc) in cases.items():
        path = ROOT / "examples" / f"{name}.tl"
        args_ = [rng.standard_normal(s).astype(np.float32) for s in shapes]
        res = {}
        for mode, flags in (("unfused", ["--no-fusion"]), ("fused", [])):
            k = tilec.compile_file(path, flags=flags)[name]
            o = np.empty(k.out_shape, np.float32)
            res[mode] = (best_time(lambda: k(*args_, out=o), args.reps * 3),
                         *kernels_and_traffic(path, flags))
        tu, nu, iu = res["unfused"]
        tf, nf, if_ = res["fused"]
        out.append(f"| `{name}` ({desc}) | {nu} / {iu / 1024:.0f} KiB | {nf} / {if_ / 1024:.0f} KiB "
                   f"| {tu * 1e6:.0f} | {tf * 1e6:.0f} | {tu / tf:.2f}x |")

    text = "\n".join(out) + "\n"
    (ROOT / "bench" / "results").mkdir(exist_ok=True)
    (ROOT / "bench" / "results" / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
