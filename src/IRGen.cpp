// AST -> tile IR: name resolution, shape inference, broadcasting checks.
#include "tilec/IR.h"
#include <map>
#include <sstream>

namespace tilec {
namespace {

std::string shapeStr(Shape s) {
  std::ostringstream os;
  os << "[" << s.r << ", " << s.c << "]";
  return os.str();
}

Shape canon(const TypeAnn &t) {
  if (t.dims.empty()) return {1, 1};
  if (t.dims.size() == 1) return {1, t.dims[0]};
  return {t.dims[0], t.dims[1]};
}

class IRGen {
public:
  explicit IRGen(const Kernel &k) : k(k) {}

  Function run() {
    f.name = k.name;
    f.tile = k.tile;
    for (const Param &p : k.params) {
      if (env.count(p.name)) throw CompileError(p.loc, "duplicate parameter '" + p.name + "'");
      Op a;
      a.kind = OpKind::Arg;
      a.argIndex = static_cast<int>(f.argNames.size());
      a.shape = canon(p.type);
      f.argNames.push_back(p.name);
      env[p.name] = emit(a);
    }
    bool returned = false;
    for (const Stmt &s : k.body) {
      if (returned) throw CompileError(s.loc, "statement after 'return' is unreachable");
      int v = gen(*s.value);
      if (s.kind == Stmt::Assign) {
        env[s.target] = v;
      } else {
        Shape want = canon(k.ret);
        if (f.ops[v].shape != want)
          throw CompileError(s.loc, "returned value has shape " + shapeStr(f.ops[v].shape) +
                                        " but kernel declares " + shapeStr(want));
        f.result = v;
        returned = true;
      }
    }
    if (!returned) throw CompileError(k.loc, "kernel '" + k.name + "' has no return statement");
    return std::move(f);
  }

private:
  const Kernel &k;
  Function f;
  std::map<std::string, int> env;

  int emit(Op op) {
    f.ops.push_back(std::move(op));
    return static_cast<int>(f.ops.size()) - 1;
  }

  int constant(float c) {
    Op op;
    op.kind = OpKind::Const;
    op.constant = c;
    op.shape = {1, 1};
    return emit(op);
  }

  int binary(OpKind kind, int a, int b, SourceLoc loc) {
    Shape s;
    if (!broadcastShapes(f.ops[a].shape, f.ops[b].shape, s))
      throw CompileError(loc, std::string("cannot broadcast ") + shapeStr(f.ops[a].shape) +
                                  " and " + shapeStr(f.ops[b].shape) + " in '" + opName(kind) +
                                  "'");
    Op op;
    op.kind = kind;
    op.operands = {a, b};
    op.shape = s;
    return emit(op);
  }

  int unary(OpKind kind, int a) {
    Op op;
    op.kind = kind;
    op.operands = {a};
    op.shape = f.ops[a].shape;
    return emit(op);
  }

  int matmul(int a, int b, SourceLoc loc) {
    Shape sa = f.ops[a].shape, sb = f.ops[b].shape;
    if (sa.c != sb.r)
      throw CompileError(loc, "matmul inner dimensions do not match: " + shapeStr(sa) + " @ " +
                                  shapeStr(sb));
    Op op;
    op.kind = OpKind::MatMul;
    op.operands = {a, b};
    op.shape = {sa.r, sb.c};
    return emit(op);
  }

  int gen(const Expr &e) {
    switch (e.kind) {
    case Expr::Num: return constant(static_cast<float>(e.num));
    case Expr::Var: {
      auto it = env.find(e.name);
      if (it == env.end()) throw CompileError(e.loc, "use of undefined name '" + e.name + "'");
      return it->second;
    }
    case Expr::Neg: return unary(OpKind::Neg, gen(*e.args[0]));
    case Expr::Binary: {
      int a = gen(*e.args[0]);
      int b = gen(*e.args[1]);
      switch (e.op) {
      case '+': return binary(OpKind::Add, a, b, e.loc);
      case '-': return binary(OpKind::Sub, a, b, e.loc);
      case '*': return binary(OpKind::Mul, a, b, e.loc);
      case '/': return binary(OpKind::Div, a, b, e.loc);
      case '@': return matmul(a, b, e.loc);
      }
      throw CompileError(e.loc, "unknown operator");
    }
    case Expr::Call: return call(e);
    }
    throw CompileError(e.loc, "unknown expression");
  }

  void arity(const Expr &e, size_t n) {
    if (e.args.size() != n)
      throw CompileError(e.loc, "'" + e.name + "' expects " + std::to_string(n) +
                                    " argument(s), got " + std::to_string(e.args.size()));
  }
  void noAxis(const Expr &e) {
    if (e.axis != -1) throw CompileError(e.loc, "'" + e.name + "' does not take an axis argument");
  }

  int reduce(OpKind kind, const Expr &e) {
    arity(e, 1);
    if (e.axis == -1)
      throw CompileError(e.loc, "'" + e.name + "' reduction requires axis=0 or axis=1");
    int a = gen(*e.args[0]);
    Op op;
    op.kind = kind;
    op.operands = {a};
    op.axis = e.axis;
    Shape s = f.ops[a].shape;
    op.shape = e.axis == 0 ? Shape{1, s.c} : Shape{s.r, 1};
    return emit(op);
  }

  int call(const Expr &e) {
    static const std::map<std::string, OpKind> unaries = {
        {"exp", OpKind::Exp},   {"log", OpKind::Log},   {"sqrt", OpKind::Sqrt},
        {"tanh", OpKind::Tanh}, {"relu", OpKind::Relu},
    };
    auto u = unaries.find(e.name);
    if (u != unaries.end()) {
      arity(e, 1);
      noAxis(e);
      return unary(u->second, gen(*e.args[0]));
    }
    if (e.name == "sum") return reduce(OpKind::ReduceSum, e);
    if ((e.name == "max" || e.name == "min") && e.axis != -1) {
      if (e.name == "min")
        throw CompileError(e.loc, "min reduction is not supported; use -max(-x, axis=...)");
      return reduce(OpKind::ReduceMax, e);
    }
    if (e.name == "max" || e.name == "min") {
      arity(e, 2);
      int a = gen(*e.args[0]);
      int b = gen(*e.args[1]);
      return binary(e.name == "max" ? OpKind::Max : OpKind::Min, a, b, e.loc);
    }
    if (e.name == "matmul") {
      arity(e, 2);
      noAxis(e);
      int a = gen(*e.args[0]);
      int b = gen(*e.args[1]);
      return matmul(a, b, e.loc);
    }
    if (e.name == "transpose") {
      arity(e, 1);
      noAxis(e);
      int a = gen(*e.args[0]);
      Op op;
      op.kind = OpKind::Transpose;
      op.operands = {a};
      op.shape = {f.ops[a].shape.c, f.ops[a].shape.r};
      return emit(op);
    }
    throw CompileError(e.loc, "unknown function '" + e.name + "'");
  }
};

} // namespace

bool broadcastShapes(Shape a, Shape b, Shape &out) {
  auto dim = [](int64_t x, int64_t y, int64_t &o) {
    if (x == y || y == 1) {
      o = x;
      return true;
    }
    if (x == 1) {
      o = y;
      return true;
    }
    return false;
  };
  return dim(a.r, b.r, out.r) && dim(a.c, b.c, out.c);
}

IRModule buildIR(const Module &m) {
  IRModule out;
  std::map<std::string, SourceLoc> seen;
  for (const Kernel &k : m.kernels) {
    if (seen.count(k.name)) throw CompileError(k.loc, "redefinition of kernel '" + k.name + "'");
    static const char *reserved[] = {"malloc", "free", "tanhf", "main", "exp",  "log",
                                     "sqrt",   "tanh", "expf",  "logf", "sqrtf"};
    for (const char *r : reserved)
      if (k.name == r)
        throw CompileError(k.loc,
                           "kernel name '" + k.name + "' clashes with a runtime/libm symbol");
    seen[k.name] = k.loc;
    out.funcs.push_back(IRGen(k).run());
  }
  return out;
}

} // namespace tilec
