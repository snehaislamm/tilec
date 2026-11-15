# RUN: %tilec %s --emit=ir-opt | %filecheck %s

kernel mlp(x: f32[16, 32], w: f32[32, 8], b: f32[8], r: f32[16, 8]) -> f32[16, 8] {
  return relu(x @ w + b) + r * 2.0
}
# Bias, ReLU and the scaled residual all land in the matmul epilogue.
# CHECK-LABEL: tile.func @mlp
# CHECK-NEXT:   %4 = tile.fused(%0, %1, %2, %3) {matmul} iter [16x8]
# CHECK:        = acc
# CHECK:        relu
# CHECK:        add
# CHECK:        tile.return %4

kernel twouse(x: f32[16, 32], w: f32[32, 8]) -> f32[16, 8] {
  h = x @ w
  return h / sum(h, axis=1)
}
# h feeds two different kernels (a reduction and a map); a matmul is never
# recomputed, so it stays a standalone kernel writing h to memory.
# CHECK-LABEL: tile.func @twouse
# CHECK:        %2 = tile.fused(%0, %1) {matmul}
# CHECK-NEXT:     ^0 = acc
# CHECK-NEXT:     yield ^0
# CHECK:        %3 = tile.fused(%2) {reduce = "sum", axis = 1}
# CHECK:        %4 = tile.fused(%2, %3) iter [16x8]
# CHECK:        tile.return %4

kernel merged(x: f32[16, 32], w: f32[32, 8]) -> f32[16, 8] {
  h = x @ w
  return relu(h) + exp(h)
}
# Both uses of h end up in one consumer after map fusion, so h becomes
# single-use and the whole expression moves into the matmul epilogue.
# CHECK-LABEL: tile.func @merged
# CHECK-NEXT:   %2 = tile.fused(%0, %1) {matmul}
# CHECK:        exp
# CHECK:        relu
# CHECK:        tile.return %2

kernel operand(x: f32[16, 32], w: f32[32, 8]) -> f32[16, 8] {
  # Producers of the matmul *operands* are not fused into the matmul.
  return exp(x) @ w
}
# CHECK-LABEL: tile.func @operand
# CHECK:        %2 = tile.fused(%0) iter [16x32]
# CHECK:        %3 = tile.fused(%2, %1) {matmul}
