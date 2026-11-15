# tilec design notes

## 1. Goals and non-goals

Goal: a compiler small enough to read in an afternoon, but structured like a
production one, with a frontend, a verified IR, independent passes behind a
pass manager, progressive lowering, a reference semantics, and tests at every
layer.

Non-goals (for now): dynamic shapes, non-f32 types, GPUs, multithreading.
Each is listed in the README roadmap with the intended approach.

## 2. Frontend

* **Lexer**: hand-written, tracks line and column for every token. `#` starts a comment.
* **Parser**: recursive descent. Precedence is `@` = `*` = `/` > `+` = `-`, matching Python. Statements need no terminator because an expression ends at the first token that cannot continue it.
* **IRGen**: lowers the AST straight to SSA. Reassigning a name rebinds it to a new value, so SSA renaming falls out for free. All shape errors are reported here with source locations, and the IR verifier treats an ill-shaped op as an internal compiler error (exit code 3), never a user error.

### Rank-2 canonicalization

Every tensor is canonicalized to rank 2 (`f32[N] → [1, N]`, scalar → `[1, 1]`).
Broadcasting then reduces to one rule per axis (size 1 broadcasts), and every
loop nest in codegen is exactly two deep (plus the matmul k loop). Higher rank
would generalize `Shape` to a small vector, and the broadcast-index
computation in `LLVMCodegen.cpp::linear` already has the right form.

## 3. The tile IR

```
tile.func @f(%0: tensor<4x8xf32>) -> tensor<4x1xf32> {
  %1 = tile.fused(%0) {reduce = "sum", axis = 1} iter [4x8] : tensor<4x1xf32> {
    ^0 = in 0
    ^1 = mul ^0, ^0
    yield ^1
  }
  tile.return %1 : tensor<4x1xf32>
}
```

* A function is a topologically ordered vector of ops, and a value id is the index of its defining op. Arguments are the leading ops. This keeps use-def walks trivial; passes that delete ops mark them `dead` and call `compact()`, which renumbers operands.
* **High-level ops** (`add`, `exp`, `matmul`, `reduce_sum`, `transpose`, …) exist only before outlining.
* **`tile.fused`** is the unit of code generation: an iteration space, an operand list, an optional matmul prologue (`operands[0] @ operands[1]` feeding the `acc` leaf), an optional reduction (sum/max over one axis), and a scalar **body**, a tiny DAG evaluated once per iteration point that reads its inputs with broadcasting. This mirrors `linalg.generic` (indexing maps + a scalar region) but restricted to the cases this compiler needs.
* **Verifier** (`IR.cpp::verify`): dominance of operands, per-op shape rules, fused-body well-formedness (nodes only reference earlier nodes, `acc` only with a matmul, inputs broadcast to the iteration space, reduction shape matches its axis). `--verify-each` runs it after every pass, and the fuzzer always does.

## 4. Passes

| Pass | Notes |
|---|---|
| `canonicalize` | Constant folding uses the same `evalUnary`/`evalBinary` as the interpreter, so folding can't change results. Identities (`x+0`, `x*1`, `x/1`, `max(x,x)`, `neg∘neg`, `relu∘relu`, `transpose∘transpose`) apply only if the result shape equals the kept operand's shape; `x[1,4] + y[3,1]*0 + 0` must not collapse to `x`. There's a lit test for exactly that. |
| `cse` | Structural hash on (kind, operands, constant bits, axis). |
| `dce` | Mark-from-result. |
| `outline` | Wraps every compute op in a one-op `tile.fused`. Constants become body leaves; repeated operands (`x * x`) are deduplicated to one input. A function that returns an argument or constant gets an explicit copy kernel, because every kernel must write `%out`. |
| `fuse` | See below. |

The cleanup passes iterate to a fixpoint (bounded at 4 rounds) before outlining.

## 5. Fusion

Consumer-driven, iterated to a fixpoint. For consumer `C` and producer `P` (defining `C`'s input `t`):

1. **Map fusion**: `P` is a pure map, has one use, and `P.shape == C.iterShape` → splice `P`'s body into `C`, substituting `P`'s yield for leaf `t`, and merge and dedupe operand lists. Works whether `C` is a map, a reduction (prologue fusion) or a matmul kernel (for `t ≥ 2`, epilogue inputs).
2. **Epilogue fusion**: `P` has a matmul and no reduction, has one use, `C` is a pure map with the same iteration space → `C` becomes `P`'s matmul with `P`'s epilogue followed by `C`'s body.
3. **Recompute fusion** (only when nothing else applies): `P` is a pure map with ≥ 2 uses, every use is a fused kernel iterating over `P.shape` (and not a matmul A/B operand), `P` is not the function result, and `bodyCost(P) ≤ budget` → inline `P` into every consumer.

Why shape equality: if `P` were `[M,1]` and `C` `[M,N]`, inlining would evaluate `P` N times per row. For a reduction result that means recomputing the whole reduction per element. Equality is a simple, conservative "no duplicated work" guarantee for rules 1 and 2. Rule 3 duplicates work on purpose, and that's what the cost model is for.

**Cost model.** `exp`/`log`/`tanh` = 8, `div`/`sqrt` = 4, others = 1. The default budget of 4 came from measurement: at 16, softmax and attention got 1.2–1.3x *slower*, because the duplicated `exp` cost more than the L2 traffic it saved. At 4, LayerNorm still recomputes `x - mean` for free and gets 1.8x faster. `--recompute-budget=N` exposes the knob.

Known gaps: no *reduction epilogue* fusion (`sum(x)/N` is a separate `[M,1]` kernel), no horizontal fusion of independent kernels over the same input, and no fusion across a transpose (it would need a non-identity indexing map in `tile.fused`).

## 6. Code generation

* **ABI**: `void k(ptr noalias nocapture readonly %arg0, …, ptr noalias nocapture %out)`. The `noalias` contract is what lets LLVM's loop vectorizer emit `<8 x float>` loops (clang prefers 256-bit vectors even on AVX-512 parts) without runtime alias checks. The generated C header documents it.
* **Loops** use alloca-held induction variables in the entry block, and `mem2reg`/SROA turn them into phis. The emitter stays a straight-line text writer with no CFG bookkeeping.
* **Matmul**: `for i0, j0 tiles { zero C-tile; for k0 { for i { a = A[i,k]; for j: C[i,j] += a * B[k,j] } }; epilogue over the C-tile }`. The innermost `j` loop streams contiguous rows of B and C (unit stride, vectorizable). The epilogue runs per tile, right after accumulation, while the tile is hot. Remainders use `smin` bounds, and odd tile sizes are covered by e2e tests and the fuzzer (`--tile=3,5,2`).
* **Floating point**: no fast-math flags, and `fmul`+`fadd` instead of `fmuladd`, so each output's accumulation order (k ascending) matches the interpreter exactly. Reductions are therefore not vectorized; enabling `reassoc` on reductions would be the next performance switch, at the cost of bitwise reproducibility.
* **Intermediates** are `malloc`ed and freed at function exit, and a fused-away tensor never gets a buffer. A liveness-based buffer reuse pass would cut peak memory.

## 7. Testing strategy

Each layer has its own oracle:

| Layer | Oracle |
|---|---|
| Parser / diagnostics | golden error text (lit, `not tilec … \| filecheck`) |
| Each pass | expected IR (lit + FileCheck), structural verifier |
| Optimizer as a whole | interpreter on frontend IR vs interpreter on optimized IR |
| Backend | interpreter vs native code; NumPy for hand-written references |
| Memory safety | ASan + UBSan on the full suite |

The fuzzer builds every sub-expression to an exact requested shape, so
matmuls, reductions on both axes, transposes and broadcasts compose freely
(matmul appears in ~80% of programs), and kernel parameters are created on
demand. A first version that picked sub-shapes at random produced matmuls in
under 3% of programs. Measuring coverage before trusting a clean run was what
caught that. The fuzzer was then mutation-tested by planting two bugs; both
were found within 60 programs.
