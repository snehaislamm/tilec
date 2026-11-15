// Producer/consumer kernel fusion over outlined `tile.fused` ops.
//
// Rules (consumer C, producer P = defining op of C's input t):
//   1. Map fusion:      P is a pure map (no matmul, no reduce), P has a single
//                       use, and P's shape equals C's iteration space.  P's
//                       body is inlined into C (this also covers reduction
//                       prologues and matmul epilogue inputs, t >= 2).
//   2. Epilogue fusion: P is a matmul kernel (no reduce) with a single use,
//                       C is a pure map over the same iteration space.  C is
//                       rewritten as P's matmul with C's body as epilogue.
// Shape equality guarantees no work is duplicated: every producer value is
// computed exactly once per consumer iteration point.
#include "tilec/Passes.h"

namespace tilec {
namespace {

struct Builder {
  Op out;
  // Find or append a deduplicated epilogue input.
  int input(int v) {
    size_t first = out.hasMatmul ? 2 : 0;
    for (size_t j = first; j < out.operands.size(); ++j)
      if (out.operands[j] == v) return static_cast<int>(j);
    out.operands.push_back(v);
    return static_cast<int>(out.operands.size()) - 1;
  }
  // Append `src`'s body, remapping its inputs through `inMap`.  If
  // `substInput >= 0`, leaves reading that input are replaced by node
  // `substNode` instead of being emitted.  Returns the new yield index.
  int append(const Op &src, const std::vector<int> &inMap, int substInput = -2,
             int substNode = -1) {
    std::vector<int> nodeMap(src.body.size(), -1);
    for (size_t b = 0; b < src.body.size(); ++b) {
      BodyNode n = src.body[b];
      if (n.kind == OpKind::Arg) {
        if (n.input == substInput) {
          nodeMap[b] = substNode;
          continue;
        }
        if (n.input != BodyNode::kAcc) n.input = inMap[n.input];
      } else if (n.kind != OpKind::Const) {
        n.a = nodeMap[n.a];
        if (isBinaryEW(n.kind)) n.b = nodeMap[n.b];
      }
      out.body.push_back(n);
      nodeMap[b] = static_cast<int>(out.body.size()) - 1;
    }
    return nodeMap.back();
  }
};

Op inlineMap(const Op &c, int t, const Op &p) {
  Builder b;
  b.out = c;
  b.out.operands.clear();
  b.out.body.clear();
  std::vector<int> cMap(c.operands.size(), -1);
  size_t first = 0;
  if (c.hasMatmul) {
    b.out.operands = {c.operands[0], c.operands[1]};
    first = 2;
  }
  for (size_t j = first; j < c.operands.size(); ++j)
    if (static_cast<int>(j) != t) cMap[j] = b.input(c.operands[j]);
  std::vector<int> pMap(p.operands.size(), -1);
  for (size_t j = 0; j < p.operands.size(); ++j)
    pMap[j] = b.input(p.operands[j]);
  int pYield = b.append(p, pMap);
  b.append(c, cMap, t, pYield);
  // C's yield may have been the substituted leaf itself (C was a copy); in
  // that case P's yield is already last.  Otherwise C's yield is last.
  return b.out;
}

Op absorbMatmul(const Op &c, int t, const Op &p) {
  Builder b;
  b.out = c;
  b.out.hasMatmul = true;
  b.out.operands = {p.operands[0], p.operands[1]};
  b.out.body.clear();
  std::vector<int> pMap(p.operands.size(), -1);
  for (size_t j = 2; j < p.operands.size(); ++j)
    pMap[j] = b.input(p.operands[j]);
  std::vector<int> cMap(c.operands.size(), -1);
  for (size_t j = 0; j < c.operands.size(); ++j)
    if (static_cast<int>(j) != t) cMap[j] = b.input(c.operands[j]);
  int pYield = b.append(p, pMap);
  b.append(c, cMap, t, pYield);
  return b.out;
}

bool isPureMap(const Op &op) {
  return op.kind == OpKind::Fused && !op.hasMatmul && op.reduce == OpKind::Arg;
}

} // namespace

int bodyCost(const Op &op) {
  int cost = 0;
  for (const BodyNode &n : op.body) {
    switch (n.kind) {
    case OpKind::Arg:
    case OpKind::Const: break;
    case OpKind::Exp:
    case OpKind::Log:
    case OpKind::Tanh:
      cost += 8; // libm call / polynomial
      break;
    case OpKind::Div:
    case OpKind::Sqrt: cost += 4; break;
    default: cost += 1;
    }
  }
  return cost;
}

/// Rule 3, recompute fusion: a cheap pure map with several consumers that all
/// iterate over its shape is duplicated into each consumer, trading a little
/// recomputation for never materializing the intermediate tensor in memory.
static bool recomputeFuse(Function &f, int budget) {
  std::vector<int> uses = computeUses(f);
  for (size_t v = f.numArgs(); v < f.ops.size(); ++v) {
    const Op &p = f.ops[v];
    if (!isPureMap(p) || p.dead || uses[v] < 2 || static_cast<int>(v) == f.result) continue;
    if (bodyCost(p) > budget) continue;
    std::vector<std::pair<size_t, int>> consumers;
    bool ok = true;
    for (size_t i = v + 1; i < f.ops.size() && ok; ++i) {
      const Op &c = f.ops[i];
      for (size_t t = 0; t < c.operands.size(); ++t) {
        if (c.operands[t] != static_cast<int>(v)) continue;
        if (c.kind != OpKind::Fused || c.iterShape != p.shape || (c.hasMatmul && t < 2))
          ok = false;
        else
          consumers.push_back({i, static_cast<int>(t)});
      }
    }
    if (!ok || static_cast<int>(consumers.size()) != uses[v]) continue;
    for (auto [i, t] : consumers)
      f.ops[i] = inlineMap(f.ops[i], t, p);
    f.ops[v].dead = true;
    compact(f);
    return true;
  }
  return false;
}

bool fuse(Function &f, int recomputeBudget) {
  bool changedAny = false;
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<int> uses = computeUses(f);
    for (size_t i = f.numArgs(); i < f.ops.size() && !changed; ++i) {
      Op &c = f.ops[i];
      if (c.kind != OpKind::Fused || c.dead) continue;
      size_t first = c.hasMatmul ? 2 : 0;
      for (size_t t = first; t < c.operands.size(); ++t) {
        int v = c.operands[t];
        Op &p = f.ops[v];
        if (p.kind != OpKind::Fused || p.dead || uses[v] != 1) continue;
        if (p.shape != c.iterShape) continue;
        if (isPureMap(p)) {
          c = inlineMap(c, static_cast<int>(t), p);
        } else if (p.hasMatmul && p.reduce == OpKind::Arg && isPureMap(c)) {
          c = absorbMatmul(c, static_cast<int>(t), p);
        } else {
          continue;
        }
        p.dead = true;
        changed = changedAny = true;
        break;
      }
    }
    if (changed) {
      compact(f);
    } else if (recomputeBudget > 0 && recomputeFuse(f, recomputeBudget)) {
      changed = changedAny = true;
    }
  }
  return changedAny;
}

} // namespace tilec
