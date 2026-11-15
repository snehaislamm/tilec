# tanh-approximation GELU: 9 elementwise ops (after constant folding) fuse into one loop.
kernel gelu(x: f32[512, 512]) -> f32[512, 512] {
  inner = 0.7978845608 * (x + 0.044715 * x * x * x)
  return 0.5 * x * (1.0 + tanh(inner))
}
