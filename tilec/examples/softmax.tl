# Numerically stable row softmax.
kernel softmax(x: f32[128, 1024]) -> f32[128, 1024] {
  m = max(x, axis=1)          # [128, 1] row max
  e = exp(x - m)              # broadcast subtract
  s = sum(e, axis=1)          # [128, 1] row sum
  return e / s
}
