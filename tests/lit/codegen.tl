# RUN: %tilec %s -o %t.ll && opt -passes=verify -disable-output %t.ll
# RUN: %tilec %s | %filecheck %s
# RUN: %tilec %s --no-tile | %filecheck %s --check-prefix=NAIVE
# RUN: %tilec %s --tile=8,16,4 | %filecheck %s --check-prefix=TILE
# RUN: %tilec %s --emit=header | %filecheck %s --check-prefix=HDR

kernel lin(x: f32[64, 32], w: f32[32, 48], b: f32[48]) -> f32[64, 48] {
  return tanh(x @ w + b)
}
# CHECK-LABEL: define void @lin(ptr noalias nocapture readonly %arg0, ptr noalias nocapture readonly %arg1, ptr noalias nocapture readonly %arg2, ptr noalias nocapture %out)
# CHECK:       ; matmul 64x32 @ 32x48, tiles 32x48x32
# CHECK:       call float @tanhf
# CHECK:       ret void
# NAIVE-NOT:   ; matmul
# TILE:        ; matmul 64x32 @ 32x48, tiles 8x16x4

@tile(16, 16, 16)
kernel attr(a: f32[32, 32], b: f32[32, 32]) -> f32[32, 32] {
  return a @ b
}
# CHECK-LABEL: define void @attr
# CHECK:       tiles 16x16x16
# TILE-LABEL:  define void @attr
# TILE:        tiles 16x16x16

kernel temps(x: f32[4, 4]) -> f32[4, 4] {
  m = sum(x, axis=1)
  return x / m
}
# Intermediates are heap-allocated and freed; the result goes straight to %out.
# CHECK-LABEL: define void @temps
# CHECK:       call ptr @malloc(i64 16)
# CHECK:       call void @free(ptr
# CHECK-NEXT:  ret void

# HDR: void lin(const float *x, const float *w, const float *b, float *out);
# HDR: void attr(const float *a, const float *b, float *out);
