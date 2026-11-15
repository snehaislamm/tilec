# Fused linear layer: matmul + bias + ReLU become a single tiled kernel
# whose epilogue runs while each output tile is still in cache.
@tile(32, 128, 128)
kernel mlp(x: f32[256, 512], w: f32[512, 128], b: f32[128]) -> f32[256, 128] {
  return relu(x @ w + b)
}
