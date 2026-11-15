# RUN: not %tilec %s 2>&1 | %filecheck %s
kernel k(x: f32[4]) -> f32[4] {
  return x + y
}
# CHECK: undefined.tl:3:14: error: use of undefined name 'y'
# CHECK-NEXT:   return x + y
# CHECK-NEXT:              ^
