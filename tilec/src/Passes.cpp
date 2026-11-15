// Canonicalization, CSE, DCE, outlining and the pass manager.
#include "tilec/Passes.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <sstream>

namespace tilec {

float evalUnary(OpKind k, float a) {
  switch (k) {
  case OpKind::Neg: return -a;
  case OpKind::Exp: return std::exp(a);
  case OpKind::Log: return std::log(a);
  case OpKind::Sqrt: return std::sqrt(a);
  case OpKind::Tanh: return std::tanh(a);
  case OpKind::Relu: return std::fmax(a, 0.0f);
  default: return a;
  }
}

float evalBinary(OpKind k, float a, float b) {
  switch (k) {
  case OpKind::Add: return a + b;
  case OpKind::Sub: return a - b;
  case OpKind::Mul: return a * b;
  case OpKind::Div: return a / b;
  case OpKind::Max: return std::fmax(a, b);
  case OpKind::Min: return std::fmin(a, b);
  default: return a;
  }
}

namespace {

void replaceAllUses(Function &f, int from, int to) {
  for (size_t i = from + 1; i < f.ops.size(); ++i)
    for (int &v : f.ops[i].operands)
      if (v == from) v = to;
  if (f.result == from) f.result = to;
}

bool isConst(const Function &f, int v, float c) {
  return f.ops[v].kind == OpKind::Const && f.ops[v].constant == c;
}

} // namespace

bool canonicalize(Function &f) {
  bool changed = false;
  for (size_t i = f.numArgs(); i < f.ops.size(); ++i) {
    Op &op = f.ops[i];
    int me = static_cast<int>(i);
    if (isBinaryEW(op.kind)) {
      int a = op.operands[0], b = op.operands[1];
      if (f.ops[a].kind == OpKind::Const && f.ops[b].kind == OpKind::Const) {
        float v = evalBinary(op.kind, f.ops[a].constant, f.ops[b].constant);
        op.kind = OpKind::Const;
        op.constant = v;
        op.operands.clear();
        changed = true;
        continue;
      }
      int keep = -1;
      switch (op.kind) {
      case OpKind::Add:
        if (isConst(f, b, 0.0f))
          keep = a;
        else if (isConst(f, a, 0.0f))
          keep = b;
        break;
      case OpKind::Sub:
        if (isConst(f, b, 0.0f)) keep = a;
        break;
      case OpKind::Mul:
        if (isConst(f, b, 1.0f))
          keep = a;
        else if (isConst(f, a, 1.0f))
          keep = b;
        break;
      case OpKind::Div:
        if (isConst(f, b, 1.0f)) keep = a;
        break;
      case OpKind::Max:
      case OpKind::Min:
        if (a == b) keep = a;
        break;
      default: break;
      }
      // Only valid if the identity does not change the broadcast shape.
      if (keep >= 0 && f.ops[keep].shape == op.shape) {
        replaceAllUses(f, me, keep);
        changed = true;
      }
      continue;
    }
    if (isUnaryEW(op.kind)) {
      int a = op.operands[0];
      if (f.ops[a].kind == OpKind::Const) {
        op.constant = evalUnary(op.kind, f.ops[a].constant);
        op.kind = OpKind::Const;
        op.operands.clear();
        changed = true;
      } else if (op.kind == OpKind::Neg && f.ops[a].kind == OpKind::Neg) {
        replaceAllUses(f, me, f.ops[a].operands[0]);
        changed = true;
      } else if (op.kind == OpKind::Relu && f.ops[a].kind == OpKind::Relu) {
        replaceAllUses(f, me, a);
        changed = true;
      }
      continue;
    }
    if (op.kind == OpKind::Transpose) {
      int a = op.operands[0];
      if (f.ops[a].kind == OpKind::Transpose) {
        replaceAllUses(f, me, f.ops[a].operands[0]);
        changed = true;
      }
    }
  }
  return changed;
}

bool cse(Function &f) {
  bool changed = false;
  std::map<std::string, int> seen;
  for (size_t i = f.numArgs(); i < f.ops.size(); ++i) {
    const Op &op = f.ops[i];
    if (op.kind == OpKind::Fused) continue; // bodies are not hashed; CSE runs before outlining
    std::ostringstream key;
    uint32_t bits;
    std::memcpy(&bits, &op.constant, sizeof bits);
    key << static_cast<int>(op.kind) << '|' << bits << '|' << op.axis;
    for (int v : op.operands)
      key << '|' << v;
    auto [it, inserted] = seen.emplace(key.str(), static_cast<int>(i));
    if (!inserted) {
      replaceAllUses(f, static_cast<int>(i), it->second);
      changed = true;
    }
  }
  return changed;
}

bool dce(Function &f) {
  std::vector<bool> live(f.ops.size(), false);
  live[f.result] = true;
  for (int i = static_cast<int>(f.ops.size()) - 1; i >= 0; --i) {
    if (!live[i]) continue;
    for (int v : f.ops[i].operands)
      live[v] = true;
  }
  bool changed = false;
  for (size_t i = f.numArgs(); i < f.ops.size(); ++i)
    if (!live[i]) {
      f.ops[i].dead = true;
      changed = true;
    }
  if (changed) compact(f);
  return changed;
}

bool outline(Function &f) {
  bool changed = false;
  for (size_t i = f.numArgs(); i < f.ops.size(); ++i) {
    Op &op = f.ops[i];
    if (op.kind == OpKind::Arg || op.kind == OpKind::Const || op.kind == OpKind::Fused ||
        op.kind == OpKind::Transpose)
      continue;
    Op fused;
    fused.kind = OpKind::Fused;
    fused.shape = op.shape;
    auto inputIdx = [&](int v) {
      for (size_t j = 0; j < fused.operands.size(); ++j)
        if (fused.operands[j] == v) return static_cast<int>(j);
      fused.operands.push_back(v);
      return static_cast<int>(fused.operands.size()) - 1;
    };
    std::map<int, int> leafOf; // operand value -> body node (dedup x * x)
    auto leaf = [&](int v) {
      auto it = leafOf.find(v);
      if (it != leafOf.end()) return it->second;
      BodyNode n;
      if (f.ops[v].kind == OpKind::Const) {
        n.kind = OpKind::Const;
        n.c = f.ops[v].constant;
      } else {
        n.kind = OpKind::Arg;
        n.input = inputIdx(v);
      }
      fused.body.push_back(n);
      return leafOf[v] = static_cast<int>(fused.body.size()) - 1;
    };
    if (op.kind == OpKind::MatMul) {
      fused.hasMatmul = true;
      fused.operands = op.operands; // positional: A, B (never deduplicated)
      fused.iterShape = op.shape;
      BodyNode acc;
      acc.input = BodyNode::kAcc;
      fused.body.push_back(acc);
    } else if (isReduce(op.kind)) {
      fused.iterShape = f.ops[op.operands[0]].shape;
      fused.reduce = op.kind;
      fused.axis = op.axis;
      leaf(op.operands[0]);
    } else {
      fused.iterShape = op.shape;
      BodyNode n;
      n.kind = op.kind;
      n.a = leaf(op.operands[0]);
      if (isBinaryEW(op.kind)) n.b = leaf(op.operands[1]);
      fused.body.push_back(n);
    }
    op = std::move(fused);
    changed = true;
  }
  // A kernel must always write its output buffer, so a function that
  // returns an argument or constant directly gets an explicit copy kernel.
  OpKind rk = f.ops[f.result].kind;
  if (rk == OpKind::Arg || rk == OpKind::Const) {
    Op copy;
    copy.kind = OpKind::Fused;
    copy.shape = copy.iterShape = f.ops[f.result].shape;
    BodyNode n;
    if (rk == OpKind::Const) {
      n.kind = OpKind::Const;
      n.c = f.ops[f.result].constant;
    } else {
      copy.operands = {f.result};
      n.input = 0;
    }
    copy.body.push_back(n);
    f.ops.push_back(std::move(copy));
    f.result = static_cast<int>(f.ops.size()) - 1;
    changed = true;
  }
  return changed;
}

int countKernels(const Function &f) {
  int n = 0;
  for (const Op &op : f.ops)
    if (op.kind == OpKind::Fused || op.kind == OpKind::Transpose) ++n;
  return n;
}

void PassManager::runOne(const std::string &name, const std::function<bool(Function &)> &p,
                         Function &f) {
  PassStat st;
  st.name = name;
  st.opsBefore = f.ops.size();
  auto t0 = std::chrono::steady_clock::now();
  st.changed = p(f);
  auto t1 = std::chrono::steady_clock::now();
  st.millis = std::chrono::duration<double, std::milli>(t1 - t0).count();
  st.opsAfter = f.ops.size();
  statsVec.push_back(st);
  if (opts.verifyEach) verify(f);
  if (opts.printAfterAll) {
    log << "// ----- IR after " << name << " (@" << f.name << ") -----\n";
    print(f, log);
  }
}

void PassManager::run(Function &f) {
  verify(f);
  if (opts.optimize) {
    // Iterate the cleanup passes to a fixpoint (bounded).
    for (int iter = 0; iter < 4; ++iter) {
      size_t before = statsVec.size();
      runOne("canonicalize", canonicalize, f);
      runOne("cse", cse, f);
      runOne("dce", dce, f);
      bool any = false;
      for (size_t i = before; i < statsVec.size(); ++i)
        any |= statsVec[i].changed;
      if (!any) break;
    }
  }
  runOne("outline", outline, f);
  if (opts.fusion) {
    int budget = opts.recomputeBudget;
    runOne("fuse", [budget](Function &fn) { return fuse(fn, budget); }, f);
  }
  runOne("dce", dce, f);
  verify(f);
}

} // namespace tilec
