# RUN: %tilec %s --emit=ir-opt --no-fusion | %filecheck %s

kernel cse(x: f32[8, 8]) -> f32[8, 8] {
  a = exp(x)
  b = exp(x)
  unused = tanh(x) * 3.0
  return a + b
}
# CHECK-LABEL: tile.func @cse
# CHECK-COUNT-1: = exp
# CHECK-NOT:    exp
# CHECK-NOT:    tanh
# CHECK:        add ^{{[0-9]+}}, ^{{[0-9]+}}
