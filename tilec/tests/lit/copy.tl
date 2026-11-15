# RUN: %tilec %s --emit=ir-opt | %filecheck %s
# RUN: %tilec %s -o %t.ll && opt -passes=verify -disable-output %t.ll

kernel ident(x: f32[3, 3]) -> f32[3, 3] {
  return x * 1.0
}
# Returning an argument still needs a kernel that writes the output buffer.
# CHECK-LABEL: tile.func @ident
# CHECK-NEXT:   %1 = tile.fused(%0) iter [3x3]
# CHECK-NEXT:     ^0 = in 0
# CHECK-NEXT:     yield ^0

kernel cst(x: f32[2, 2]) -> f32 {
  return 3.0 * 2.0
}
# CHECK-LABEL: tile.func @cst
# CHECK:          ^0 = const 6
