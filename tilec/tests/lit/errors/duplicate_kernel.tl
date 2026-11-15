# RUN: not %tilec %s 2>&1 | %filecheck %s
kernel k(x: f32[4]) -> f32[4] {
  return x
}
kernel k(x: f32[4]) -> f32[4] {
  return x
}
# CHECK: error: redefinition of kernel 'k'
