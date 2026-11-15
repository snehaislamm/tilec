# RUN: not %tilec %s 2>&1 | %filecheck %s
kernel k(x: f16[4]) -> f32[4] {
  return x
}
# CHECK: error: unsupported element type 'f16' (only f32 is supported)
