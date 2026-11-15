#pragma once
#include "tilec/IR.h"
#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace tilec {

/// Scalar semantics shared by constant folding, the interpreter, and (by
/// construction) the LLVM backend, so all three agree bit-for-bit where the
/// hardware allows.
float evalUnary(OpKind k, float a);
float evalBinary(OpKind k, float a, float b);

// ---- individual passes; each returns true if it changed the IR ----

/// Constant folding + algebraic identities (x+0, x*1, neg(neg x), ...).
bool canonicalize(Function &f);
/// Common-subexpression elimination over structurally identical ops.
bool cse(Function &f);
/// Remove ops not reachable from the function result.
bool dce(Function &f);
/// Wrap every compute op in a single-op `tile.fused` kernel.  Required
/// before codegen; fusion then merges these kernels.
bool outline(Function &f);
/// Producer/consumer fusion: elementwise chains, reduction prologues,
/// matmul epilogues and (if recomputeBudget > 0) recompute fusion of cheap
/// multi-consumer maps.  See Fusion.cpp for the exact legality rules.
bool fuse(Function &f, int recomputeBudget = 4);
/// Static cost estimate of a fused body (used by the recompute heuristic).
int bodyCost(const Op &op);

struct PassOptions {
  bool fusion = true;
  int recomputeBudget = 4; // 0 disables; 4 admits cheap ops, not exp/log/tanh
  bool optimize = true;     // canonicalize/cse/dce
  bool verifyEach = false;
  bool printAfterAll = false;
  bool timePasses = false;
};

struct PassStat {
  std::string name;
  double millis = 0.0;
  bool changed = false;
  size_t opsBefore = 0, opsAfter = 0;
};

class PassManager {
public:
  explicit PassManager(PassOptions opts, std::ostream &log) : opts(opts), log(log) {}
  void run(Function &f);
  const std::vector<PassStat> &stats() const { return statsVec; }

private:
  void runOne(const std::string &name, const std::function<bool(Function &)> &p, Function &f);
  PassOptions opts;
  std::ostream &log;
  std::vector<PassStat> statsVec;
};

/// Number of loop-nest kernels the function will lower to.
int countKernels(const Function &f);

} // namespace tilec
