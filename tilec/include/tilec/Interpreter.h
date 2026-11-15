#pragma once
#include "tilec/IR.h"
#include <vector>

namespace tilec {

struct Tensor {
  Shape shape;
  std::vector<float> data; // row-major
};

/// Reference interpreter for tile IR at any stage of the pipeline (before or
/// after outlining/fusion).  Used as the oracle for differential testing of
/// the optimizer and the LLVM backend.
Tensor interpret(const Function &f, const std::vector<Tensor> &args);

} // namespace tilec
