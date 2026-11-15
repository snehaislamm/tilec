# LayerNorm over the last dimension with learned scale and shift.
kernel layernorm(x: f32[256, 768], g: f32[768], b: f32[768]) -> f32[256, 768] {
  mu = sum(x, axis=1) / 768.0
  d = x - mu
  var = sum(d * d, axis=1) / 768.0
  return d / sqrt(var + 0.00001) * g + b
}
