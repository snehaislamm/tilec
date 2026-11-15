# RUN: %tilec %s --emit=ir-opt --no-fusion | %filecheck %s
# RUN: %tilec %s --emit=ir-opt -O0 --no-fusion | %filecheck %s --check-prefix=O0

kernel fold(x: f32[2, 2]) -> f32[2, 2] {
  k = 2.0 * 0.5          # folds to 1.0, then x * 1.0 -> x
  y = (x * k) + 0.0      # + 0.0 -> identity
  z = -(-y)              # double negation
  return exp(z)
}
# CHECK-LABEL: tile.func @fold
# CHECK-NEXT:   %1 = tile.fused(%0) iter [2x2]
# CHECK-NEXT:     ^0 = in 0
# CHECK-NEXT:     ^1 = exp ^0
# CHECK-NOT:    mul
# CHECK:        tile.return %1

# O0-LABEL: tile.func @fold
# O0:         ^{{[0-9]+}} = mul
# O0:         ^{{[0-9]+}} = neg

kernel tt(x: f32[3, 5]) -> f32[3, 5] {
  return relu(relu(transpose(transpose(x))))
}
# CHECK-LABEL: tile.func @tt
# CHECK-NOT:    transpose
# CHECK:        ^1 = relu ^0
# CHECK-NEXT:   yield ^1

kernel noshape(x: f32[1, 4], y: f32[3, 1]) -> f32[3, 4] {
  # x + 0 must NOT be simplified to x when the zero broadcasts the result.
  return x + y * 0.0 + 0.0
}
# CHECK-LABEL: tile.func @noshape
# CHECK:        mul
