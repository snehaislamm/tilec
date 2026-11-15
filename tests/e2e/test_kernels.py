"""End-to-end tests: compile real kernels to native code and compare against NumPy.

Every example is checked under several pipeline configurations so that the
optimizer (fusion on/off, -O0) and the backend (tiled vs naive matmul, odd tile
sizes that exercise remainder loops) are each validated independently.
"""
from pathlib import Path

import numpy as np
import pytest

import tilec

ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / "examples"
RNG = np.random.default_rng(1234)

CONFIGS = {
    "default": [],
    "no-fusion": ["--no-fusion"],
    "no-tile": ["--no-tile"],
    "O0": ["-O0"],
    "odd-tiles": ["--tile=7,13,5"],
    "no-recompute": ["--recompute-budget=0"],
}


def randn(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def softmax_ref(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


CASES = {
    "mlp": (lambda: [randn(256, 512), randn(512, 128), randn(128)],
            lambda x, w, b: np.maximum(x @ w + b, 0), 2e-3),
    "softmax": (lambda: [randn(128, 1024)], softmax_ref, 1e-6),
    "gelu": (lambda: [randn(512, 512)],
             lambda x: 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))), 1e-5),
    "attention": (lambda: [randn(128, 64), randn(128, 64), randn(128, 64)],
                  lambda q, k, v: softmax_ref(q @ k.T * 0.125) @ v, 1e-4),
    "layernorm": (lambda: [randn(256, 768), randn(768), randn(768)],
                  lambda x, g, b: (x - x.mean(1, keepdims=True))
                  / np.sqrt(x.var(1, keepdims=True) + 1e-5) * g + b, 1e-4),
}


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("name", list(CASES))
def test_example_matches_numpy(name, config):
    make, ref, tol = CASES[name]
    args = make()
    k = tilec.compile_file(EX / f"{name}.tl", flags=CONFIGS[config])[name]
    got = k(*args)
    want = ref(*args).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=tol, atol=tol)


@pytest.mark.parametrize("name", list(CASES))
def test_interpreter_stages_agree(name):
    """Frontend IR, optimized IR and native code must agree."""
    make, ref, tol = CASES[name]
    args = make()
    k = tilec.compile_file(EX / f"{name}.tl")[name]
    native = k(*args)
    pre = tilec.interpret(EX / f"{name}.tl", name, args, k.out_shape, stage="input")
    post = tilec.interpret(EX / f"{name}.tl", name, args, k.out_shape, stage="opt")
    np.testing.assert_allclose(pre, post, rtol=tol, atol=tol)
    np.testing.assert_allclose(native, post, rtol=tol, atol=tol)


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (37, 53, 29), (64, 1, 64), (5, 300, 3), (130, 70, 257)])
@pytest.mark.parametrize("flags", [[], ["--no-tile"], ["--tile=16,16,16"]])
def test_matmul_shapes(m, k, n, flags):
    src = f"""
kernel mm(a: f32[{m}, {k}], b: f32[{k}, {n}], c: f32[{n}]) -> f32[{m}, {n}] {{
  return a @ b - c
}}
"""
    fn = tilec.compile_source(src, flags=flags)["mm"]
    a, b, c = randn(m, k), randn(k, n), randn(n)
    np.testing.assert_allclose(fn(a, b, c), a @ b - c, rtol=1e-4, atol=1e-4)


def test_reductions_and_broadcast():
    src = """
kernel red(x: f32[33, 17], col: f32[33, 1], row: f32[17]) -> f32[1, 17] {
  y = exp(x * 0.1) * col + row
  return max(y, axis=0) - sum(y, axis=0) / 33.0
}
"""
    fn = tilec.compile_source(src)["red"]
    x, col, row = randn(33, 17), randn(33, 1), randn(17)
    y = np.exp(x * 0.1) * col + row
    want = y.max(0, keepdims=True) - y.sum(0, keepdims=True) / 33.0
    np.testing.assert_allclose(fn(x, col, row), want, rtol=1e-5, atol=1e-5)


def test_constant_matmul_operand():
    # A scalar constant used directly as a matmul operand must be materialized.
    src = """
kernel cm(b: f32[1, 6]) -> f32[1, 6] {
  return 2.5 @ b
}
"""
    b = randn(1, 6)
    np.testing.assert_allclose(tilec.compile_source(src)["cm"](b), 2.5 * b, rtol=1e-6)


def test_multiple_kernels_in_one_module():
    src = """
kernel a(x: f32[4, 4]) -> f32[4, 4] { return x + 1.0 }
kernel b(x: f32[4, 4]) -> f32[4, 4] { return transpose(x) * 2.0 }
"""
    ks = tilec.compile_source(src)
    x = randn(4, 4)
    np.testing.assert_allclose(ks["a"](x), x + 1)
    np.testing.assert_allclose(ks["b"](x), x.T * 2)


def test_wrong_arity_raises():
    k = tilec.compile_file(EX / "softmax.tl")["softmax"]
    with pytest.raises(TypeError):
        k()
    with pytest.raises(ValueError):
        k(randn(3, 3))


def test_compile_error_surfaces():
    with pytest.raises(tilec.CompileError, match="undefined name"):
        tilec.compile_source("kernel k(x: f32[2]) -> f32[2] { return y }")
