# RUN: not %tilec %s 2>&1 | %filecheck %s
kernel k(x: f32[2, 2, 2]) -> f32[2] {
  return x
}
# CHECK: error: tensors of rank > 2 are not supported
