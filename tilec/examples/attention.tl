# Single-head scaled dot-product attention: softmax(Q K^T / sqrt(d)) V
kernel attention(q: f32[128, 64], k: f32[128, 64], v: f32[128, 64]) -> f32[128, 64] {
  s = (q @ transpose(k)) * 0.125      # 1/sqrt(64), fused into the QK^T epilogue
  m = max(s, axis=1)
  p = exp(s - m)
  l = sum(p, axis=1)
  return (p / l) @ v
}
