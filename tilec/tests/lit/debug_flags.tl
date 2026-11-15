# RUN: %tilec %s --emit=ir-opt --print-ir-after-all --verify-each 2>&1 >/dev/null | %filecheck %s
# RUN: %tilec %s --emit=ir-opt --time-passes 2>&1 >/dev/null | %filecheck %s --check-prefix=TIME

kernel k(x: f32[4, 4]) -> f32[4, 4] {
  return exp(x) + 1.0
}
# CHECK: // ----- IR after canonicalize (@k) -----
# CHECK: // ----- IR after cse (@k) -----
# CHECK: // ----- IR after outline (@k) -----
# CHECK: // ----- IR after fuse (@k) -----
# TIME: // pass timing for @k
# TIME: fuse
# TIME: total
