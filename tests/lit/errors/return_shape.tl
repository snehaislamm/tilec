# RUN: not %tilec %s 2>&1 | %filecheck %s
kernel k(x: f32[4, 3]) -> f32[4, 4] {
  return x
}
# CHECK: error: returned value has shape [4, 3] but kernel declares [4, 4]
