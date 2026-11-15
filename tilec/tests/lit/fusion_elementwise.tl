# RUN: %tilec %s --emit=ir-opt | %filecheck %s
# RUN: %tilec %s --emit=ir-opt --no-fusion | %filecheck %s --check-prefix=NOFUSE
# RUN: %tilec %s --emit=ir-opt --stats 2>&1 >/dev/null | %filecheck %s --check-prefix=STATS

kernel gelu(x: f32[64, 64]) -> f32[64, 64] {
  inner = 0.7978845608 * (x + 0.044715 * x * x * x)
  return 0.5 * x * (1.0 + tanh(inner))
}
# One loop nest; x is read once (deduplicated input).
# CHECK-LABEL: tile.func @gelu
# CHECK:        %1 = tile.fused(%0) iter [64x64]
# CHECK:        tanh
# CHECK:        tile.return %1
# NOFUSE-COUNT-9: tile.fused
# STATS: @gelu: 9 kernel(s) unfused -> 1 kernel(s) after pipeline
