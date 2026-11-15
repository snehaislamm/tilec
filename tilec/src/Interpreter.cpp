#include "tilec/Interpreter.h"
#include "tilec/Passes.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tilec {
namespace {

inline float load(const Tensor &t, int64_t i, int64_t j) {
  int64_t r = t.shape.r > 1 ? i : 0;
  int64_t c = t.shape.c > 1 ? j : 0;
  return t.data[r * t.shape.c + c];
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  Tensor out{{a.shape.r, b.shape.c}, {}};
  out.data.assign(out.shape.numel(), 0.0f);
  int64_t M = a.shape.r, K = a.shape.c, N = b.shape.c;
  // Same per-element accumulation order (k ascending) as the generated code.
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) {
      float av = a.data[i * K + k];
      for (int64_t j = 0; j < N; ++j)
        out.data[i * N + j] = out.data[i * N + j] + av * b.data[k * N + j];
    }
  return out;
}

float evalBody(const Op &op, const std::vector<const Tensor *> &ins, const Tensor *acc, int64_t i,
               int64_t j, std::vector<float> &scratch) {
  scratch.resize(op.body.size());
  for (size_t b = 0; b < op.body.size(); ++b) {
    const BodyNode &n = op.body[b];
    float v;
    if (n.kind == OpKind::Arg)
      v = n.input == BodyNode::kAcc ? acc->data[i * acc->shape.c + j] : load(*ins[n.input], i, j);
    else if (n.kind == OpKind::Const)
      v = n.c;
    else if (isBinaryEW(n.kind))
      v = evalBinary(n.kind, scratch[n.a], scratch[n.b]);
    else
      v = evalUnary(n.kind, scratch[n.a]);
    scratch[b] = v;
  }
  return scratch.back();
}

} // namespace

Tensor interpret(const Function &f, const std::vector<Tensor> &args) {
  if (static_cast<int>(args.size()) != f.numArgs())
    throw std::runtime_error("interpret: expected " + std::to_string(f.numArgs()) + " arguments");
  std::vector<Tensor> vals(f.ops.size());
  std::vector<float> scratch;
  for (size_t i = 0; i < f.ops.size(); ++i) {
    const Op &op = f.ops[i];
    Tensor &out = vals[i];
    out.shape = op.shape;
    auto in = [&](int k) -> const Tensor & { return vals[op.operands[k]]; };
    switch (op.kind) {
    case OpKind::Arg:
      if (args[op.argIndex].shape != op.shape ||
          args[op.argIndex].data.size() != static_cast<size_t>(op.shape.numel()))
        throw std::runtime_error("interpret: argument " + std::to_string(op.argIndex) +
                                 " has the wrong shape");
      out = args[op.argIndex];
      break;
    case OpKind::Const: out.data = {op.constant}; break;
    case OpKind::MatMul: out = matmul(in(0), in(1)); break;
    case OpKind::Transpose: {
      const Tensor &a = in(0);
      out.data.resize(out.shape.numel());
      for (int64_t r = 0; r < a.shape.r; ++r)
        for (int64_t c = 0; c < a.shape.c; ++c)
          out.data[c * a.shape.r + r] = a.data[r * a.shape.c + c];
      break;
    }
    case OpKind::ReduceSum:
    case OpKind::ReduceMax: {
      const Tensor &a = in(0);
      bool isMax = op.kind == OpKind::ReduceMax;
      float init = isMax ? -std::numeric_limits<float>::infinity() : 0.0f;
      out.data.assign(out.shape.numel(), init);
      for (int64_t r = 0; r < a.shape.r; ++r)
        for (int64_t c = 0; c < a.shape.c; ++c) {
          float &dst = out.data[op.axis == 1 ? r : c];
          float v = a.data[r * a.shape.c + c];
          dst = isMax ? std::fmax(dst, v) : dst + v;
        }
      break;
    }
    case OpKind::Fused: {
      std::vector<const Tensor *> ins;
      for (size_t k = 0; k < op.operands.size(); ++k)
        ins.push_back(&in(static_cast<int>(k)));
      Tensor acc;
      if (op.hasMatmul) acc = matmul(in(0), in(1));
      Shape it = op.iterShape;
      if (op.reduce == OpKind::Arg) {
        out.data.resize(out.shape.numel());
        for (int64_t r = 0; r < it.r; ++r)
          for (int64_t c = 0; c < it.c; ++c)
            out.data[r * it.c + c] = evalBody(op, ins, &acc, r, c, scratch);
      } else {
        bool isMax = op.reduce == OpKind::ReduceMax;
        float init = isMax ? -std::numeric_limits<float>::infinity() : 0.0f;
        out.data.assign(out.shape.numel(), init);
        for (int64_t r = 0; r < it.r; ++r)
          for (int64_t c = 0; c < it.c; ++c) {
            float v = evalBody(op, ins, &acc, r, c, scratch);
            float &dst = out.data[op.axis == 1 ? r : c];
            dst = isMax ? std::fmax(dst, v) : dst + v;
          }
      }
      break;
    }
    default: {
      out.data.resize(out.shape.numel());
      for (int64_t r = 0; r < op.shape.r; ++r)
        for (int64_t c = 0; c < op.shape.c; ++c) {
          float v = isBinaryEW(op.kind) ? evalBinary(op.kind, load(in(0), r, c), load(in(1), r, c))
                                        : evalUnary(op.kind, load(in(0), r, c));
          out.data[r * op.shape.c + c] = v;
        }
    }
    }
  }
  return vals[f.result];
}

} // namespace tilec
