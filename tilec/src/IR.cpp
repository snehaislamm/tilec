// IR utilities: printing, verification, use counting, compaction.
#include "tilec/IR.h"
#include <cstdio>
#include <stdexcept>

namespace tilec {

const char *opName(OpKind k) {
  switch (k) {
  case OpKind::Arg: return "arg";
  case OpKind::Const: return "const";
  case OpKind::Add: return "add";
  case OpKind::Sub: return "sub";
  case OpKind::Mul: return "mul";
  case OpKind::Div: return "div";
  case OpKind::Max: return "max";
  case OpKind::Min: return "min";
  case OpKind::Neg: return "neg";
  case OpKind::Exp: return "exp";
  case OpKind::Log: return "log";
  case OpKind::Sqrt: return "sqrt";
  case OpKind::Tanh: return "tanh";
  case OpKind::Relu: return "relu";
  case OpKind::MatMul: return "matmul";
  case OpKind::ReduceSum: return "reduce_sum";
  case OpKind::ReduceMax: return "reduce_max";
  case OpKind::Transpose: return "transpose";
  case OpKind::Fused: return "fused";
  }
  return "?";
}

bool isBinaryEW(OpKind k) {
  switch (k) {
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
  case OpKind::Max:
  case OpKind::Min: return true;
  default: return false;
  }
}

bool isUnaryEW(OpKind k) {
  switch (k) {
  case OpKind::Neg:
  case OpKind::Exp:
  case OpKind::Log:
  case OpKind::Sqrt:
  case OpKind::Tanh:
  case OpKind::Relu: return true;
  default: return false;
  }
}

namespace {

std::string ty(Shape s) {
  return "tensor<" + std::to_string(s.r) + "x" + std::to_string(s.c) + "xf32>";
}

std::string fmtFloat(float v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v));
  return buf;
}

void fail(const Function &f, int op, const std::string &msg) {
  throw std::logic_error("verifier: @" + f.name + " op %" + std::to_string(op) + ": " + msg);
}

} // namespace

void print(const Function &f, std::ostream &os) {
  os << "tile.func @" << f.name << "(";
  for (int i = 0; i < f.numArgs(); ++i) {
    if (i) os << ", ";
    os << "%" << i << ": " << ty(f.ops[i].shape);
  }
  os << ") -> " << ty(f.ops[f.result].shape);
  if (!f.tile.empty())
    os << " attributes {tile = [" << f.tile[0] << ", " << f.tile[1] << ", " << f.tile[2] << "]}";
  os << " {\n";
  for (size_t i = f.numArgs(); i < f.ops.size(); ++i) {
    const Op &op = f.ops[i];
    os << "  %" << i << " = tile." << opName(op.kind);
    if (op.kind == OpKind::Const) {
      os << " " << fmtFloat(op.constant) << " : " << ty(op.shape) << "\n";
      continue;
    }
    if (op.kind == OpKind::Fused) os << "(";
    for (size_t j = 0; j < op.operands.size(); ++j)
      os << (j ? ", " : (op.kind == OpKind::Fused ? "" : " ")) << "%" << op.operands[j];
    if (op.kind == OpKind::Fused) {
      os << ")";
      std::vector<std::string> attrs;
      if (op.hasMatmul) attrs.push_back("matmul");
      if (op.reduce != OpKind::Arg)
        attrs.push_back(std::string("reduce = \"") +
                        (op.reduce == OpKind::ReduceSum ? "sum" : "max") +
                        "\", axis = " + std::to_string(op.axis));
      if (!attrs.empty()) {
        os << " {";
        for (size_t j = 0; j < attrs.size(); ++j)
          os << (j ? ", " : "") << attrs[j];
        os << "}";
      }
      os << " iter [" << op.iterShape.r << "x" << op.iterShape.c << "] : " << ty(op.shape)
         << " {\n";
      for (size_t b = 0; b < op.body.size(); ++b) {
        const BodyNode &n = op.body[b];
        os << "    ^" << b << " = ";
        if (n.kind == OpKind::Arg) {
          if (n.input == BodyNode::kAcc)
            os << "acc";
          else
            os << "in " << n.input;
        } else if (n.kind == OpKind::Const) {
          os << "const " << fmtFloat(n.c);
        } else if (isBinaryEW(n.kind)) {
          os << opName(n.kind) << " ^" << n.a << ", ^" << n.b;
        } else {
          os << opName(n.kind) << " ^" << n.a;
        }
        os << "\n";
      }
      os << "    yield ^" << op.body.size() - 1 << "\n  }\n";
      continue;
    }
    if (isReduce(op.kind)) os << " {axis = " << op.axis << "}";
    os << " : " << ty(op.shape) << "\n";
  }
  os << "  tile.return %" << f.result << " : " << ty(f.ops[f.result].shape) << "\n";
  os << "}\n";
}

void print(const IRModule &m, std::ostream &os) {
  for (size_t i = 0; i < m.funcs.size(); ++i) {
    if (i) os << "\n";
    print(m.funcs[i], os);
  }
}

void verify(const Function &f) {
  int n = static_cast<int>(f.ops.size());
  if (f.result < 0 || f.result >= n)
    throw std::logic_error("verifier: @" + f.name + ": invalid result value");
  for (int i = 0; i < n; ++i) {
    const Op &op = f.ops[i];
    if (op.dead) fail(f, i, "op still marked dead");
    bool isArgPos = i < f.numArgs();
    if (isArgPos != (op.kind == OpKind::Arg))
      fail(f, i, "arguments must be exactly the leading ops");
    if (op.kind == OpKind::Arg && op.argIndex != i) fail(f, i, "argument index mismatch");
    for (int v : op.operands)
      if (v < 0 || v >= i)
        fail(f, i, "operand %" + std::to_string(v) + " does not dominate its use");
    if (op.shape.r <= 0 || op.shape.c <= 0) fail(f, i, "non-positive shape");
    auto sh = [&](int j) { return f.ops[op.operands[j]].shape; };
    auto want = [&](size_t cnt) {
      if (op.operands.size() != cnt)
        fail(f, i, std::string(opName(op.kind)) + " expects " + std::to_string(cnt) + " operands");
    };
    Shape s;
    switch (op.kind) {
    case OpKind::Arg:
    case OpKind::Const:
      want(0);
      if (op.kind == OpKind::Const && op.shape != Shape{1, 1}) fail(f, i, "constants must be 1x1");
      break;
    case OpKind::MatMul:
      want(2);
      if (sh(0).c != sh(1).r || op.shape != Shape{sh(0).r, sh(1).c})
        fail(f, i, "matmul shape mismatch");
      break;
    case OpKind::ReduceSum:
    case OpKind::ReduceMax:
      want(1);
      if (op.axis == 0 ? op.shape != Shape{1, sh(0).c}
                       : (op.axis == 1 ? op.shape != Shape{sh(0).r, 1} : true))
        fail(f, i, "reduction shape mismatch");
      break;
    case OpKind::Transpose:
      want(1);
      if (op.shape != Shape{sh(0).c, sh(0).r}) fail(f, i, "transpose shape mismatch");
      break;
    case OpKind::Fused: {
      size_t first = 0;
      if (op.hasMatmul) {
        if (op.operands.size() < 2) fail(f, i, "matmul-fused op needs A and B operands");
        if (sh(0).c != sh(1).r || op.iterShape != Shape{sh(0).r, sh(1).c})
          fail(f, i, "fused matmul shape mismatch");
        if (op.reduce != OpKind::Arg) fail(f, i, "matmul fusion with a reduction is not supported");
        first = 2;
      }
      for (size_t j = first; j < op.operands.size(); ++j) {
        Shape o;
        if (!broadcastShapes(op.iterShape, sh(static_cast<int>(j)), o) || o != op.iterShape)
          fail(f, i, "fused input does not broadcast to the iteration space");
      }
      if (op.reduce == OpKind::Arg) {
        if (op.shape != op.iterShape) fail(f, i, "map result shape must equal iteration space");
      } else {
        Shape exp = op.axis == 0 ? Shape{1, op.iterShape.c} : Shape{op.iterShape.r, 1};
        if (op.axis < 0 || op.axis > 1 || op.shape != exp)
          fail(f, i, "fused reduction shape mismatch");
      }
      if (op.body.empty()) fail(f, i, "empty fused body");
      for (size_t b = 0; b < op.body.size(); ++b) {
        const BodyNode &nd = op.body[b];
        int bi = static_cast<int>(b);
        if (nd.kind == OpKind::Arg) {
          if (nd.input == BodyNode::kAcc) {
            if (!op.hasMatmul) fail(f, i, "acc used without matmul");
          } else if (nd.input < static_cast<int>(first) ||
                     nd.input >= static_cast<int>(op.operands.size())) {
            fail(f, i, "body leaf references invalid input");
          }
        } else if (nd.kind == OpKind::Const) {
        } else if (isBinaryEW(nd.kind)) {
          if (nd.a < 0 || nd.a >= bi || nd.b < 0 || nd.b >= bi)
            fail(f, i, "body node references a later node");
        } else if (isUnaryEW(nd.kind)) {
          if (nd.a < 0 || nd.a >= bi) fail(f, i, "body node references a later node");
        } else {
          fail(f, i, "invalid body node kind");
        }
      }
      break;
    }
    default:
      if (isBinaryEW(op.kind)) {
        want(2);
        if (!broadcastShapes(sh(0), sh(1), s) || s != op.shape)
          fail(f, i, "elementwise broadcast mismatch");
      } else if (isUnaryEW(op.kind)) {
        want(1);
        if (sh(0) != op.shape) fail(f, i, "unary shape mismatch");
      } else {
        fail(f, i, "unknown op kind");
      }
    }
  }
}

std::vector<int> computeUses(const Function &f) {
  std::vector<int> uses(f.ops.size(), 0);
  for (const Op &op : f.ops) {
    if (op.dead) continue;
    for (int v : op.operands)
      ++uses[v];
  }
  if (f.result >= 0) ++uses[f.result];
  return uses;
}

void compact(Function &f) {
  std::vector<int> remap(f.ops.size(), -1);
  std::vector<Op> kept;
  for (size_t i = 0; i < f.ops.size(); ++i) {
    if (f.ops[i].dead) continue;
    remap[i] = static_cast<int>(kept.size());
    kept.push_back(std::move(f.ops[i]));
  }
  for (Op &op : kept)
    for (int &v : op.operands) {
      if (remap[v] < 0) throw std::logic_error("compact: live op uses a dead value");
      v = remap[v];
    }
  f.result = remap[f.result];
  f.ops = std::move(kept);
}

} // namespace tilec
