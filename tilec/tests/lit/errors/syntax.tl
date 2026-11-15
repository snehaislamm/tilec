# RUN: not %tilec %s 2>&1 | %filecheck %s
kernel k(x: f32[4]) -> f32[4] {
  return (x + 1.0
}
# CHECK: error: expected ')', found '}'
