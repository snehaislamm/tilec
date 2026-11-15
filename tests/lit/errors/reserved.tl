# RUN: not %tilec %s 2>&1 | %filecheck %s
kernel malloc(x: f32[4]) -> f32[4] {
  return x
}
# CHECK: error: kernel name 'malloc' clashes with a runtime/libm symbol
