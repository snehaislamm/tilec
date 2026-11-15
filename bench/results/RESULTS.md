# tilec benchmark results

CPU: Intel(R) Xeon(R) Processor @ 2.30GHz, single thread. clang -O3 -march=native -fveclib=libmvec -lmvec. Best of 7 runs. NumPy 2.5.3 (OPENBLAS_NUM_THREADS=1).


## 1. Matmul 512x512x512

| implementation | time (ms) | GFLOP/s | speedup vs naive |
|---|---:|---:|---:|
| naive i-j-k (`--no-tile`) | 108.66 | 2.5 | 1.0x |
| tiled (default 32x256x128) | 6.93 | 38.8 | 15.7x |
| NumPy / OpenBLAS (hand-tuned asm) | 1.80 | 148.9 | 60.3x |

## 2. Tile-size sweep (matmul 512^3)

| TM x TN x TK | C tile + B panel (KiB) | GFLOP/s |
|---|---:|---:|
| 8 x 64 x 64 | 18 | 33.4 |
| 16 x 128 x 64 | 40 | 37.2 |
| 32 x 128 x 128 | 80 | 37.4 |
| 32 x 256 x 128 | 160 | 39.0 |
| 64 x 256 x 64 | 128 | 37.2 |
| 64 x 512 x 256 | 640 | 20.9 |
| 128 x 128 x 128 | 128 | 34.2 |
| 16 x 512 x 512 | 1056 | 20.9 |

Best: 32x256x128 at 39.0 GFLOP/s.

## 3. Kernel fusion

| kernel | unfused: loop nests / intermediates | fused: loop nests / intermediates | unfused (us) | fused (us) | speedup |
|---|---:|---:|---:|---:|---:|
| `mlp` (relu(x @ w + b)) | 3 / 256 KiB | 1 / 0 KiB | 834 | 849 | 0.98x |
| `gelu` (tanh-GELU, 9 elementwise ops) | 9 / 8192 KiB | 1 / 0 KiB | 5007 | 4741 | 1.06x |
| `softmax` (row softmax) | 5 / 1025 KiB | 4 / 513 KiB | 398 | 397 | 1.00x |
| `layernorm` (LayerNorm) | 11 / 3078 KiB | 5 / 4 KiB | 423 | 233 | 1.81x |
| `attention` (softmax(QK^T/8)V) | 9 / 353 KiB | 7 / 225 KiB | 133 | 130 | 1.03x |
