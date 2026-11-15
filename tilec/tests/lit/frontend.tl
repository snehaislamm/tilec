# RUN: %tilec %s --emit=ir | %filecheck %s
# Rank-1 and scalar types are canonicalized to rank 2; broadcasting infers shapes.

kernel f(x: f32[4, 8], b: f32[8], s: f32) -> f32[4, 8] {
  y = x * s + b
  return -y
}
# CHECK-LABEL: tile.func @f(%0: tensor<4x8xf32>, %1: tensor<1x8xf32>, %2: tensor<1x1xf32>) -> tensor<4x8xf32> {
# CHECK-NEXT:   %3 = tile.mul %0, %2 : tensor<4x8xf32>
# CHECK-NEXT:   %4 = tile.add %3, %1 : tensor<4x8xf32>
# CHECK-NEXT:   %5 = tile.neg %4 : tensor<4x8xf32>
# CHECK-NEXT:   tile.return %5 : tensor<4x8xf32>
