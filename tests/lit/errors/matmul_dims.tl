# RUN: not %tilec %s 2>&1 | %filecheck %s
kernel k(a: f32[4, 3], b: f32[4, 3]) -> f32[4, 3] {
  return a @ b
}
# CHECK: error: matmul inner dimensions do not match: [4, 3] @ [4, 3]
