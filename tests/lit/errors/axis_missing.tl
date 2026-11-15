# RUN: not %tilec %s 2>&1 | %filecheck %s
kernel k(x: f32[4, 4]) -> f32[4, 1] {
  return sum(x)
}
# CHECK: error: 'sum' reduction requires axis=0 or axis=1
