#pragma once
#include "tilec/AST.h"
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace tilec {

/// Every tensor is canonicalized to rank 2: f32[N] -> [1, N], f32 -> [1, 1].
/// Broadcasting follows NumPy rules on size-1 dimensions.
struct Shape {
  int64_t r = 1, c = 1;
  int64_t numel() const { return r * c; }
  bool operator==(const Shape &o) const { return r == o.r && c == o.c; }
  bool operator!=(const Shape &o) const { return !(*this == o); }
};

enum class OpKind {
  Arg,
  Const,
  // elementwise binary
  Add,
  Sub,
  Mul,
  Div,
  Max,
  Min,
  // elementwise unary
  Neg,
  Exp,
  Log,
  Sqrt,
  Tanh,
  Relu,
  // structured
  MatMul,
  ReduceSum,
  ReduceMax,
  Transpose,
  // produced by the outlining/fusion passes
  Fused,
};

const char *opName(OpKind k);
bool isBinaryEW(OpKind k);
bool isUnaryEW(OpKind k);
inline bool isElementwise(OpKind k) {
  return isBinaryEW(k) || isUnaryEW(k);
}
inline bool isReduce(OpKind k) {
  return k == OpKind::ReduceSum || k == OpKind::ReduceMax;
}

/// One scalar operation inside a fused kernel body.  The body is evaluated
/// once per point of the iteration space and reads its inputs with
/// broadcasting.  Leaves: `kind == Arg` with `input` = index into the fused
/// op's operand list, or `input == kAcc` for the matmul accumulator.
struct BodyNode {
  static constexpr int kAcc = -1;
  OpKind kind = OpKind::Arg;
  int a = -1, b = -1; // indices of earlier body nodes
  int input = 0;      // for Arg leaves
  float c = 0.0f;     // for Const leaves
};

struct Op {
  OpKind kind = OpKind::Arg;
  std::vector<int> operands; // value ids (= op indices in Function::ops)
  Shape shape;               // result shape
  float constant = 0.0f;     // Const
  int axis = -1;             // ReduceSum / ReduceMax (and fused reductions)
  int argIndex = -1;         // Arg

  // ---- Fused only ----
  bool hasMatmul = false;      // operands[0] @ operands[1] feeds acc
  OpKind reduce = OpKind::Arg; // Arg = no reduction, else ReduceSum/Max
  Shape iterShape;             // iteration space of the body
  std::vector<BodyNode> body;  // last node is the yielded value

  bool dead = false; // scratch flag used by passes
};

struct Function {
  std::string name;
  std::vector<std::string> argNames;
  std::vector<Op> ops; // topologically ordered; first N ops are Args
  int result = -1;
  std::vector<int64_t> tile; // optional per-kernel tile sizes (m, n, k)

  int numArgs() const { return static_cast<int>(argNames.size()); }
};

struct IRModule {
  std::vector<Function> funcs;
};

/// Lower the AST to tile IR, performing name resolution, shape inference and
/// broadcasting checks.
IRModule buildIR(const Module &m);

/// Textual MLIR-style printing.
void print(const IRModule &m, std::ostream &os);
void print(const Function &f, std::ostream &os);

/// Structural verifier; throws std::logic_error describing the first
/// violation.  Run after every pass with --verify-each.
void verify(const Function &f);

/// Number of uses of each value (including the function result).
std::vector<int> computeUses(const Function &f);

/// Remove ops flagged `dead` and renumber operands.
void compact(Function &f);

/// Broadcast two shapes; returns false if incompatible.
bool broadcastShapes(Shape a, Shape b, Shape &out);

} // namespace tilec
