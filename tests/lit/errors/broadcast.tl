# RUN: not %tilec %s 2>&1 | %filecheck %s
kernel k(x: f32[4, 3], y: f32[4, 2]) -> f32[4, 3] {
  return x + y
}
# CHECK: error: cannot broadcast [4, 3] and [4, 2] in 'add'
