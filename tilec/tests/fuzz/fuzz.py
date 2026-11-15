#!/usr/bin/env python3
"""Differential fuzzer for tilec.

Generates random, well-typed kernels (elementwise chains, broadcasting,
reductions, matmuls, transposes, reused variables) and checks that

    interpret(frontend IR)  ==  interpret(optimized IR)  ==  native code

for a randomly chosen pipeline configuration per program.  The frontend
interpreter is the oracle: it executes the program exactly as written,
before any pass touches it.  Failing programs are written to
tests/fuzz/failures/ with the exact repro command.
"""
import argparse
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import tilec  # noqa: E402

CONFIGS = [[], ["--no-fusion"], ["--no-tile"], ["-O0"], ["--tile=3,5,2"],
           ["--recompute-budget=0"], ["--recompute-budget=100"], ["-O0", "--no-fusion"]]


class Gen:
    """Builds every sub-expression to an exact requested shape, so matmuls,
    reductions, transposes and broadcasts compose freely.  Kernel parameters
    are created on demand when no existing value has the needed shape."""

    def __init__(self, rng: random.Random):
        self.r = rng
        self.lines = []
        self.vars = []    # (name, shape) usable as leaves
        self.params = []  # (name, shape)
        self.n = 0

    def const(self):
        return f"{self.r.choice([0.5, 1.0, 2.0, 0.25, 3.0, 0.0, 1.5, -1.0])}"

    def param(self, shape):
        name = f"a{len(self.params)}"
        self.params.append((name, shape))
        self.vars.append((name, shape))
        return name

    def leaf(self, shape):
        if shape == (1, 1) and self.r.random() < 0.5:
            return self.const()
        cands = [n for n, s in self.vars if s == shape]
        if cands and self.r.random() < 0.85:
            return self.r.choice(cands)
        return self.param(shape)

    def sub_shape(self, shape):
        """A shape that broadcasts to `shape`."""
        R, C = shape
        return self.r.choice([(R, C), (R, C), (R, 1), (1, C), (1, 1)])

    def dim(self):
        return self.r.randint(1, 13)

    def exact(self, shape, depth):
        R, C = shape
        if depth <= 0:
            return self.leaf(shape)
        x = self.r.random()
        if x < 0.28:
            op = self.r.choice(["+", "-", "*"])
            a = self.exact(shape, depth - 1)
            b = self.exact(self.sub_shape(shape), depth - 1)
            if self.r.random() < 0.5:
                a, b = b, a
            return f"({a} {op} {b})"
        if x < 0.36:
            f = self.r.choice(["max", "min"])
            return f"{f}({self.exact(shape, depth - 1)}, {self.exact(self.sub_shape(shape), depth - 1)})"
        if x < 0.52:
            a = self.exact(shape, depth - 1)
            f = self.r.choice(["tanh", "relu", "neg", "exp", "div", "sqrt"])
            return {"neg": f"(-{a})",
                    "exp": f"exp(tanh({a}))",
                    "div": f"({a} / ({a} * {a} + 1.0))",
                    "sqrt": f"sqrt({a} * {a} + 0.5)"}.get(f, f"{f}({a})")
        if x < 0.66 and (R == 1 or C == 1):
            f = self.r.choice(["sum", "max"])
            if C == 1:
                return f"{f}({self.exact((R, self.dim()), depth - 1)}, axis=1)"
            return f"{f}({self.exact((self.dim(), C), depth - 1)}, axis=0)"
        if x < 0.80:
            K = self.dim()
            return f"({self.exact((R, K), depth - 1)} @ {self.exact((K, C), depth - 1)})"
        if x < 0.87:
            return f"transpose({self.exact((C, R), depth - 1)})"
        return self.leaf(shape)

    def program(self):
        R, C = self.r.randint(1, 17), self.r.randint(1, 17)
        for i in range(self.r.randint(0, 4)):
            tgt = self.sub_shape((R, C)) if self.r.random() < 0.5 else (self.dim(), self.dim())
            e = self.exact(tgt, self.r.randint(1, 3))
            name = f"v{i}"
            self.lines.append(f"  {name} = {e}")
            self.vars.append((name, tgt))
        e = self.exact((R, C), self.r.randint(1, 4))
        if not self.params:
            self.param((R, C))
        ps = ", ".join(f"{n}: f32[{s[0]}, {s[1]}]" for n, s in self.params)
        body = "\n".join(self.lines + [f"  return {e}"])
        return f"kernel fz({ps}) -> f32[{R}, {C}] {{\n{body}\n}}\n", self.params, (R, C)


def close(a, b):
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    scale = max(1.0, float(np.nanmax(np.abs(b))) if b.size and np.isfinite(b).any() else 1.0)
    ok = np.isclose(a, b, rtol=1e-4, atol=1e-4 * scale) | both_nan
    return bool(ok.all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilec", default=None)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.tilec:
        import os
        os.environ["TILEC"] = args.tilec

    work = Path(tempfile.mkdtemp(prefix="tilec_fuzz_"))
    faildir = ROOT / "tests" / "fuzz" / "failures"
    failures = 0
    for it in range(args.iters):
        seed = args.seed * 1_000_003 + it
        rng = random.Random(seed)
        src, params, out_shape = Gen(rng).program()
        path = work / f"p{it}.tl"
        path.write_text(src)
        nprng = np.random.default_rng(seed)
        inputs = [nprng.standard_normal(s).astype(np.float32) for _, s in params]
        cfg = rng.choice(CONFIGS)
        stage = "?"
        try:
            stage = "interp-input"
            oracle = tilec.interpret(path, "fz", inputs, out_shape, stage="input")
            stage = "interp-opt"
            opt = tilec.interpret(path, "fz", inputs, out_shape, stage="opt",
                                  flags=cfg + ["--verify-each"])
            stage = "native"
            native = tilec.compile_file(path, flags=cfg, workdir=str(work))["fz"](*inputs)
            bad = None
            if not close(opt, oracle):
                bad = "optimized IR disagrees with frontend IR"
            elif not close(native, oracle):
                bad = "native code disagrees with frontend IR"
        except Exception as e:  # compiler crash, verifier failure, bad codegen
            bad = f"{stage} failed: {e}"
        if bad:
            failures += 1
            faildir.mkdir(exist_ok=True)
            fp = faildir / f"seed{seed}.tl"
            fp.write_text(f"# {bad}\n# flags: {' '.join(cfg)}\n" + src)
            print(f"[{it}] FAIL ({' '.join(cfg) or 'default'}): {bad}\n  saved {fp}")
        elif (it + 1) % 50 == 0:
            print(f"[{it + 1}/{args.iters}] ok")
    print(f"\nfuzz: {args.iters - failures}/{args.iters} programs agreed across all stages")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
