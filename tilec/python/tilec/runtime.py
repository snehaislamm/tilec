"""Compile tilec kernels to a shared library and call them through ctypes.

Pipeline:  .tl --tilec--> .ll --clang -O3--> .so --ctypes--> NumPy
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_HEADER_RE = re.compile(r"^// (\w+): (.*)-> out\[(\d+)x(\d+)\]$")
_ARG_RE = re.compile(r"(\w+)\[(\d+)x(\d+)\]")


class CompileError(RuntimeError):
    pass


def tilec_binary() -> str:
    env = os.environ.get("TILEC")
    if env:
        return env
    for cand in (_REPO / "build" / "tilec", _REPO / "build" / "Release" / "tilec"):
        if cand.exists():
            return str(cand)
    found = shutil.which("tilec")
    if found:
        return found
    raise FileNotFoundError("tilec binary not found; build it or set $TILEC")


def _clang() -> str:
    return os.environ.get("TILEC_CLANG", "clang")


def default_cflags() -> tuple:
    """-O3 for the host CPU; on x86-64 Linux also vectorize exp/log/tanh via glibc libmvec."""
    flags = ["-O3", "-march=native"]
    if sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64") \
            and os.environ.get("TILEC_VECLIB", "1") != "0":
        flags += ["-fveclib=libmvec", "-lmvec"]
    return tuple(flags)


def _run(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CompileError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


@dataclass
class Kernel:
    name: str
    arg_names: List[str]
    arg_shapes: List[Tuple[int, int]]
    out_shape: Tuple[int, int]
    _fn: object
    _lib: object  # keep the library alive

    def __call__(self, *args: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if len(args) != len(self.arg_shapes):
            raise TypeError(f"{self.name} takes {len(self.arg_shapes)} arrays, got {len(args)}")
        bufs = []
        for name, shape, a in zip(self.arg_names, self.arg_shapes, args):
            a = np.ascontiguousarray(a, dtype=np.float32)
            if a.size != shape[0] * shape[1]:
                raise ValueError(f"{self.name}: argument '{name}' has {a.size} elements, "
                                 f"expected {shape[0]}x{shape[1]}")
            bufs.append(a)
        if out is None:
            out = np.empty(self.out_shape, dtype=np.float32)
        ptrs = [b.ctypes.data_as(ctypes.c_void_p) for b in bufs]
        self._fn(*ptrs, out.ctypes.data_as(ctypes.c_void_p))
        return out


def _parse_header(text: str) -> Dict[str, Tuple[List[str], List[Tuple[int, int]], Tuple[int, int]]]:
    sigs = {}
    for line in text.splitlines():
        m = _HEADER_RE.match(line.strip())
        if not m:
            continue
        name, args, r, c = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        arg_names, arg_shapes = [], []
        for am in _ARG_RE.finditer(args):
            arg_names.append(am.group(1))
            arg_shapes.append((int(am.group(2)), int(am.group(3))))
        sigs[name] = (arg_names, arg_shapes, (r, c))
    return sigs


def compile_file(path: str | os.PathLike, flags: Sequence[str] = (),
                 cflags: Sequence[str] | None = None,
                 workdir: str | None = None) -> Dict[str, Kernel]:
    """Compile every kernel in `path`; returns {name: Kernel}."""
    path = str(path)
    work = Path(workdir or tempfile.mkdtemp(prefix="tilec_"))
    stem = Path(path).stem
    ll = work / f"{stem}.ll"
    so = work / f"{stem}.so"
    tilec = tilec_binary()
    _run([tilec, path, "--emit=llvm", "-o", str(ll), *flags])
    header = _run([tilec, path, "--emit=header", *flags])
    cflags = default_cflags() if cflags is None else tuple(cflags)
    _run([_clang(), *cflags, "-Wno-override-module", "-Wno-invalid-feature-combination",
          "-shared", "-fPIC", str(ll), "-o", str(so), "-lm"])
    lib = ctypes.CDLL(str(so))
    kernels = {}
    for name, (arg_names, arg_shapes, out_shape) in _parse_header(header).items():
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p] * (len(arg_shapes) + 1)
        kernels[name] = Kernel(name, arg_names, arg_shapes, out_shape, fn, lib)
    return kernels


def compile_source(src: str, **kw) -> Dict[str, Kernel]:
    work = tempfile.mkdtemp(prefix="tilec_")
    p = Path(work) / "kernel.tl"
    p.write_text(src)
    return compile_file(p, workdir=work, **kw)


def interpret(path: str | os.PathLike, kernel: str, inputs: Sequence[np.ndarray],
              out_shape: Tuple[int, int], stage: str = "opt",
              flags: Sequence[str] = ()) -> np.ndarray:
    """Run the C++ reference interpreter on the IR at `stage` (input|opt)."""
    work = Path(tempfile.mkdtemp(prefix="tilec_interp_"))
    cmd = [tilec_binary(), str(path), "--interp", "--kernel", kernel,
           f"--stage={stage}", *flags]
    for i, a in enumerate(inputs):
        f = work / f"in{i}.bin"
        np.ascontiguousarray(a, dtype=np.float32).tofile(f)
        cmd += ["--in", str(f)]
    out = work / "out.bin"
    cmd += ["--out", str(out)]
    _run(cmd)
    return np.fromfile(out, dtype=np.float32).reshape(out_shape)
