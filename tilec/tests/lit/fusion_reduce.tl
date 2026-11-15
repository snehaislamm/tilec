# RUN: %tilec %s --emit=ir-opt | %filecheck %s
# RUN: %tilec %s --emit=ir-opt --recompute-budget=16 | %filecheck %s --check-prefix=RECOMP

kernel sumsq(x: f32[32, 128]) -> f32[32, 1] {
  return sum(x * x, axis=1)
}
# CHECK-LABEL: tile.func @sumsq
# CHECK-NEXT:   %1 = tile.fused(%0) {reduce = "sum", axis = 1} iter [32x128] : tensor<32x1xf32>
# CHECK-NEXT:     ^0 = in 0
# CHECK-NEXT:     ^1 = mul ^0, ^0

kernel colmax(x: f32[32, 128]) -> f32[1, 128] {
  return max(exp(x), axis=0)
}
# CHECK-LABEL: tile.func @colmax
# CHECK:        {reduce = "max", axis = 0} iter [32x128] : tensor<1x128xf32>

kernel softmax(x: f32[8, 64]) -> f32[8, 64] {
  m = max(x, axis=1)
  e = exp(x - m)
  return e / sum(e, axis=1)
}
# exp(x - m) has two consumers.  Its body cost (sub=1 + exp=8) exceeds the
# default recompute budget of 4, so it is materialized once (benchmarks show
# recomputing exp is slower than the extra memory traffic).
# CHECK-LABEL: tile.func @softmax
# CHECK:        %1 = tile.fused(%0) {reduce = "max", axis = 1}
# CHECK:        %2 = tile.fused(%0, %1) iter [8x64]
# CHECK:        exp
# CHECK:        %3 = tile.fused(%2) {reduce = "sum", axis = 1}
# CHECK:        %4 = tile.fused(%2, %3) iter [8x64]
# CHECK:        tile.return %4
# With a budget of 16 the exp is duplicated into both consumers instead, and
# the 8x64 intermediate is never written.  A reduced [8,1] value is never
# fused into a [8,64] consumer (that would recompute the reduction per element).
# RECOMP-LABEL: tile.func @softmax
# RECOMP:        %1 = tile.fused(%0) {reduce = "max", axis = 1}
# RECOMP:        %2 = tile.fused(%0, %1) {reduce = "sum", axis = 1}
# RECOMP:        exp
# RECOMP:        %3 = tile.fused({{.*}}) iter [8x64]
# RECOMP:        exp
# RECOMP:        div
# RECOMP:        tile.return %3

kernel center(x: f32[8, 64]) -> f32[8, 64] {
  d = x - sum(x, axis=1) / 64.0
  return d * d + d
}
# d = x - mean has three uses inside one consumer after map fusion, so the
# [8,64] centered tensor is never materialized.  The `/ 64.0` runs on the
# [8,1] reduction result and stays a separate tiny kernel (no reduction
# epilogue fusion yet).
# CHECK-LABEL: tile.func @center
# CHECK:        %1 = tile.fused(%0) {reduce = "sum", axis = 1}
# CHECK:        %2 = tile.fused(%1) iter [8x1]
# CHECK:        %3 = tile.fused(%0, %2) iter [8x64]
# CHECK:        sub
# CHECK:        tile.return %3
