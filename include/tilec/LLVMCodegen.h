#pragma once
#include "tilec/IR.h"
#include <cstdint>
#include <ostream>

namespace tilec {

struct CodegenOptions {
  bool tileMatmul = true;
  // Default tile sizes (m, n, k); a kernel's @tile(...) attribute overrides.
  int64_t tileM = 32, tileN = 256, tileK = 128;
};

/// Lower fused tile IR (after `outline`, optionally `fuse`) to textual LLVM IR.
///
/// ABI per kernel:  void @name(ptr %arg0, ..., ptr %argN-1, ptr %out)
/// All buffers are dense row-major f32; inputs are read-only and no buffer
/// may alias another.
void emitLLVM(const IRModule &m, const CodegenOptions &opts, std::ostream &os);

/// C prototypes for the generated kernels.
void emitHeader(const IRModule &m, std::ostream &os);

} // namespace tilec
