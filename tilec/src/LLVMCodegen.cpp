// Lowering of fused tile IR to textual LLVM IR.
//
// Each `tile.fused` op becomes one loop nest.  Loop counters live in allocas
// in the entry block (promoted to SSA phis by mem2reg/SROA in `opt`), which
// keeps this emitter simple while producing IR that -O3 vectorizes.
#include "tilec/LLVMCodegen.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>

namespace tilec {
namespace {

std::string floatLit(float v) {
  // LLVM IR spells float constants as the hex bits of the equivalent double.
  double d = static_cast<double>(v);
  uint64_t bits;
  std::memcpy(&bits, &d, sizeof bits);
  char buf[32];
  std::snprintf(buf, sizeof buf, "0x%016llX", static_cast<unsigned long long>(bits));
  return buf;
}

bool isLit(const std::string &s) {
  return !s.empty() && (std::isdigit(static_cast<unsigned char>(s[0])) || s[0] == '-');
}

class FnEmitter {
public:
  std::ostringstream allocas, code;

  std::string fresh(const char *hint = "t") {
    return std::string("%") + hint + std::to_string(nextVal++);
  }
  std::string label(const char *hint) { return std::string(hint) + std::to_string(nextLabel++); }
  void inst(const std::string &s) { code << "  " << s << "\n"; }
  std::string def(const std::string &rhs, const char *hint = "t") {
    std::string v = fresh(hint);
    inst(v + " = " + rhs);
    return v;
  }
  void block(const std::string &l) { code << l << ":\n"; }

  // --- i64 index arithmetic with trivial constant folding ---
  std::string add(const std::string &a, const std::string &b) {
    if (a == "0") return b;
    if (b == "0") return a;
    if (isLit(a) && isLit(b)) return std::to_string(std::stoll(a) + std::stoll(b));
    return def("add nsw i64 " + a + ", " + b, "i");
  }
  std::string mul(const std::string &a, int64_t c) {
    if (c == 0 || a == "0") return "0";
    if (c == 1) return a;
    if (isLit(a)) return std::to_string(std::stoll(a) * c);
    return def("mul nsw i64 " + a + ", " + std::to_string(c), "i");
  }
  std::string smin(const std::string &a, const std::string &b) {
    std::string c = def("icmp slt i64 " + a + ", " + b, "c");
    return def("select i1 " + c + ", i64 " + a + ", i64 " + b, "i");
  }
  std::string gep(const std::string &ptr, const std::string &idx) {
    return def("getelementptr inbounds float, ptr " + ptr + ", i64 " + idx, "p");
  }
  std::string load(const std::string &ptr, const std::string &idx) {
    return def("load float, ptr " + gep(ptr, idx) + ", align 4", "v");
  }
  void store(const std::string &val, const std::string &ptr, const std::string &idx) {
    inst("store float " + val + ", ptr " + gep(ptr, idx) + ", align 4");
  }
  std::string allocaFloat() {
    std::string v = fresh("acc");
    allocas << "  " << v << " = alloca float, align 4\n";
    return v;
  }

  /// for (iv = lo; iv < hi; iv += step) body(iv)
  void loop(const std::string &lo, const std::string &hi, int64_t step, const char *name,
            const std::function<void(const std::string &)> &body) {
    std::string slot = fresh(name);
    allocas << "  " << slot << ".addr = alloca i64, align 8\n";
    std::string lc = label((std::string(name) + ".cond").c_str());
    std::string lb = label((std::string(name) + ".body").c_str());
    std::string le = label((std::string(name) + ".end").c_str());
    inst("store i64 " + lo + ", ptr " + slot + ".addr, align 8");
    inst("br label %" + lc);
    block(lc);
    std::string iv = def("load i64, ptr " + slot + ".addr, align 8", name);
    std::string c = def("icmp slt i64 " + iv + ", " + hi, "c");
    inst("br i1 " + c + ", label %" + lb + ", label %" + le);
    block(lb);
    body(iv);
    std::string cur = def("load i64, ptr " + slot + ".addr, align 8", "i");
    std::string nx = def("add nsw i64 " + cur + ", " + std::to_string(step), "i");
    inst("store i64 " + nx + ", ptr " + slot + ".addr, align 8");
    inst("br label %" + lc);
    block(le);
  }

private:
  int nextVal = 0, nextLabel = 0;
};

class KernelLowering {
public:
  KernelLowering(const Function &f, const CodegenOptions &o) : f(f), opts(o) {}

  void run(std::ostream &os) {
    std::vector<std::string> ptr(f.ops.size());
    for (int i = 0; i < f.numArgs(); ++i)
      ptr[i] = "%arg" + std::to_string(i);
    std::vector<std::string> temps;
    for (size_t i = f.numArgs(); i < f.ops.size(); ++i) {
      const Op &op = f.ops[i];
      if (static_cast<int>(i) == f.result) {
        ptr[i] = "%out";
      } else if (op.kind == OpKind::Const) {
        // Only constants feeding matmul/transpose operands survive DCE.
        std::string a = e.fresh("cst");
        e.allocas << "  " << a << " = alloca float, align 4\n";
        e.inst("store float " + floatLit(op.constant) + ", ptr " + a + ", align 4");
        ptr[i] = a;
      } else {
        int64_t bytes = op.shape.numel() * 4;
        ptr[i] = e.def("call ptr @malloc(i64 " + std::to_string(bytes) + ")", "buf");
        temps.push_back(ptr[i]);
      }
    }
    for (size_t i = f.numArgs(); i < f.ops.size(); ++i) {
      const Op &op = f.ops[i];
      if (op.kind == OpKind::Const) continue;
      std::vector<std::string> ins;
      for (int v : op.operands)
        ins.push_back(ptr[v]);
      e.code << "  ; kernel %" << i << " = tile." << opName(op.kind) << "\n";
      if (op.kind == OpKind::Transpose)
        lowerTranspose(op, ins[0], ptr[i]);
      else if (op.kind == OpKind::Fused)
        lowerFused(op, ins, ptr[i]);
      else
        throw std::logic_error(std::string("codegen: unlowered op tile.") + opName(op.kind) +
                               " (run outline first)");
    }
    for (const std::string &t : temps)
      e.inst("call void @free(ptr " + t + ")");
    e.inst("ret void");

    os << "define void @" << f.name << "(";
    for (int i = 0; i < f.numArgs(); ++i)
      os << "ptr noalias nocapture readonly %arg" << i << ", ";
    os << "ptr noalias nocapture %out) #0 {\nentry:\n";
    os << e.allocas.str() << e.code.str() << "}\n";
  }

private:
  const Function &f;
  const CodegenOptions &opts;
  FnEmitter e;

  Shape shapeOf(int v) const { return f.ops[v].shape; }

  std::string linear(Shape s, const std::string &i, const std::string &j) {
    std::string r = s.r > 1 ? i : "0";
    std::string c = s.c > 1 ? j : "0";
    return e.add(e.mul(r, s.c), c);
  }

  /// Emit the scalar body at point (i, j); returns the yielded SSA value.
  std::string emitBody(const Op &op, const std::vector<std::string> &ins, const std::string &i,
                       const std::string &j, const std::function<std::string()> &acc) {
    std::vector<std::string> val(op.body.size());
    std::map<int, std::string> loaded;
    std::string accVal;
    for (size_t b = 0; b < op.body.size(); ++b) {
      const BodyNode &n = op.body[b];
      if (n.kind == OpKind::Arg) {
        if (n.input == BodyNode::kAcc) {
          if (accVal.empty()) accVal = acc();
          val[b] = accVal;
        } else {
          auto it = loaded.find(n.input);
          if (it == loaded.end()) {
            Shape s = shapeOf(op.operands[n.input]);
            it = loaded.emplace(n.input, e.load(ins[n.input], linear(s, i, j))).first;
          }
          val[b] = it->second;
        }
        continue;
      }
      if (n.kind == OpKind::Const) {
        val[b] = floatLit(n.c);
        continue;
      }
      const std::string &a = val[n.a];
      switch (n.kind) {
      case OpKind::Add: val[b] = e.def("fadd float " + a + ", " + val[n.b]); break;
      case OpKind::Sub: val[b] = e.def("fsub float " + a + ", " + val[n.b]); break;
      case OpKind::Mul: val[b] = e.def("fmul float " + a + ", " + val[n.b]); break;
      case OpKind::Div: val[b] = e.def("fdiv float " + a + ", " + val[n.b]); break;
      case OpKind::Max:
        val[b] = e.def("call float @llvm.maxnum.f32(float " + a + ", float " + val[n.b] + ")");
        break;
      case OpKind::Min:
        val[b] = e.def("call float @llvm.minnum.f32(float " + a + ", float " + val[n.b] + ")");
        break;
      case OpKind::Neg: val[b] = e.def("fneg float " + a); break;
      case OpKind::Exp: val[b] = e.def("call float @llvm.exp.f32(float " + a + ")"); break;
      case OpKind::Log: val[b] = e.def("call float @llvm.log.f32(float " + a + ")"); break;
      case OpKind::Sqrt: val[b] = e.def("call float @llvm.sqrt.f32(float " + a + ")"); break;
      case OpKind::Tanh: val[b] = e.def("call float @tanhf(float " + a + ")"); break;
      case OpKind::Relu:
        val[b] = e.def("call float @llvm.maxnum.f32(float " + a + ", float 0.000000e+00)");
        break;
      default: throw std::logic_error("codegen: bad body node");
      }
    }
    return val.back();
  }

  std::string combine(OpKind red, const std::string &a, const std::string &b) {
    if (red == OpKind::ReduceSum) return e.def("fadd float " + a + ", " + b);
    return e.def("call float @llvm.maxnum.f32(float " + a + ", float " + b + ")");
  }

  void lowerTranspose(const Op &op, const std::string &in, const std::string &out) {
    int64_t R = op.shape.c, C = op.shape.r; // input is R x C
    e.loop("0", std::to_string(R), 1, "r", [&](const std::string &r) {
      e.loop("0", std::to_string(C), 1, "c", [&](const std::string &c) {
        std::string v = e.load(in, e.add(e.mul(r, C), c));
        e.store(v, out, e.add(e.mul(c, R), r));
      });
    });
  }

  void lowerFused(const Op &op, const std::vector<std::string> &ins, const std::string &out) {
    if (op.hasMatmul) return lowerMatmul(op, ins, out);
    int64_t R = op.iterShape.r, C = op.iterShape.c;
    auto noAcc = []() -> std::string { throw std::logic_error("no acc"); };
    std::string sR = std::to_string(R), sC = std::to_string(C);
    if (op.reduce == OpKind::Arg) {
      e.loop("0", sR, 1, "i", [&](const std::string &i) {
        e.loop("0", sC, 1, "j", [&](const std::string &j) {
          e.store(emitBody(op, ins, i, j, noAcc), out, e.add(e.mul(i, C), j));
        });
      });
      return;
    }
    std::string init = op.reduce == OpKind::ReduceSum ? "0.000000e+00" : "0xFFF0000000000000";
    if (op.axis == 1) {
      std::string acc = e.allocaFloat();
      e.loop("0", sR, 1, "i", [&](const std::string &i) {
        e.inst("store float " + init + ", ptr " + acc + ", align 4");
        e.loop("0", sC, 1, "j", [&](const std::string &j) {
          std::string v = emitBody(op, ins, i, j, noAcc);
          std::string cur = e.def("load float, ptr " + acc + ", align 4");
          e.inst("store float " + combine(op.reduce, cur, v) + ", ptr " + acc + ", align 4");
        });
        std::string fin = e.def("load float, ptr " + acc + ", align 4");
        e.store(fin, out, i);
      });
    } else {
      e.loop("0", sC, 1, "j", [&](const std::string &j) { e.store(init, out, j); });
      e.loop("0", sR, 1, "i", [&](const std::string &i) {
        e.loop("0", sC, 1, "j", [&](const std::string &j) {
          std::string v = emitBody(op, ins, i, j, noAcc);
          std::string cur = e.load(out, j);
          e.store(combine(op.reduce, cur, v), out, j);
        });
      });
    }
  }

  void lowerMatmul(const Op &op, const std::vector<std::string> &ins, const std::string &out) {
    int64_t M = op.iterShape.r, N = op.iterShape.c, K = shapeOf(op.operands[0]).c;
    const std::string &A = ins[0], &B = ins[1];
    bool trivialEpilogue =
        op.body.size() == 1 && op.body[0].kind == OpKind::Arg && op.body[0].input == BodyNode::kAcc;
    std::string sM = std::to_string(M), sN = std::to_string(N), sK = std::to_string(K);
    std::string zero = "0.000000e+00";

    if (!opts.tileMatmul) {
      // Naive i-j-k loop nest: scalar dot product per output element.
      std::string acc = e.allocaFloat();
      e.loop("0", sM, 1, "i", [&](const std::string &i) {
        e.loop("0", sN, 1, "j", [&](const std::string &j) {
          e.inst("store float " + zero + ", ptr " + acc + ", align 4");
          e.loop("0", sK, 1, "k", [&](const std::string &k) {
            std::string a = e.load(A, e.add(e.mul(i, K), k));
            std::string b = e.load(B, e.add(e.mul(k, N), j));
            std::string p = e.def("fmul float " + a + ", " + b);
            std::string cur = e.def("load float, ptr " + acc + ", align 4");
            e.inst("store float " + e.def("fadd float " + cur + ", " + p) + ", ptr " + acc +
                   ", align 4");
          });
          auto accFn = [&]() { return e.def("load float, ptr " + acc + ", align 4"); };
          std::string v = emitBody(op, ins, i, j, accFn);
          e.store(v, out, e.add(e.mul(i, N), j));
        });
      });
      return;
    }

    int64_t TM = opts.tileM, TN = opts.tileN, TK = opts.tileK;
    if (f.tile.size() == 3) {
      TM = f.tile[0];
      TN = f.tile[1];
      TK = f.tile[2];
    }
    TM = std::min(TM, M);
    TN = std::min(TN, N);
    TK = std::min(TK, K);
    e.code << "  ; matmul " << M << "x" << K << " @ " << K << "x" << N << ", tiles " << TM << "x"
           << TN << "x" << TK << "\n";
    // Cache-blocked i0/j0/k0 tiles with an i-k-j micro-kernel: the innermost
    // j loop streams contiguous rows of B and C, which LLVM vectorizes.
    e.loop("0", sM, TM, "i0", [&](const std::string &i0) {
      std::string i1 = e.smin(e.add(i0, std::to_string(TM)), sM);
      e.loop("0", sN, TN, "j0", [&](const std::string &j0) {
        std::string j1 = e.smin(e.add(j0, std::to_string(TN)), sN);
        e.loop(i0, i1, 1, "i", [&](const std::string &i) {
          e.loop(j0, j1, 1, "j",
                 [&](const std::string &j) { e.store(zero, out, e.add(e.mul(i, N), j)); });
        });
        e.loop("0", sK, TK, "k0", [&](const std::string &k0) {
          std::string k1 = e.smin(e.add(k0, std::to_string(TK)), sK);
          e.loop(i0, i1, 1, "i", [&](const std::string &i) {
            std::string rowA = e.mul(i, K);
            std::string rowC = e.mul(i, N);
            e.loop(k0, k1, 1, "k", [&](const std::string &k) {
              std::string a = e.load(A, e.add(rowA, k));
              std::string rowB = e.mul(k, N);
              e.loop(j0, j1, 1, "j", [&](const std::string &j) {
                std::string b = e.load(B, e.add(rowB, j));
                std::string cIdx = e.add(rowC, j);
                std::string c = e.load(out, cIdx);
                std::string p = e.def("fmul float " + a + ", " + b);
                e.store(e.def("fadd float " + c + ", " + p), out, cIdx);
              });
            });
          });
        });
        if (!trivialEpilogue) {
          // Fused epilogue applied while the C tile is still hot in cache.
          e.loop(i0, i1, 1, "i", [&](const std::string &i) {
            e.loop(j0, j1, 1, "j", [&](const std::string &j) {
              std::string idx = e.add(e.mul(i, N), j);
              auto accFn = [&]() { return e.load(out, idx); };
              e.store(emitBody(op, ins, i, j, accFn), out, idx);
            });
          });
        }
      });
    });
  }
};

} // namespace

void emitLLVM(const IRModule &m, const CodegenOptions &opts, std::ostream &os) {
  os << "; ModuleID = 'tilec'\nsource_filename = \"tilec\"\n\n";
  for (const Function &f : m.funcs) {
    KernelLowering(f, opts).run(os);
    os << "\n";
  }
  os << "declare ptr @malloc(i64)\n"
        "declare void @free(ptr)\n"
        "declare float @tanhf(float) #1\n"
        "declare float @llvm.exp.f32(float)\n"
        "declare float @llvm.log.f32(float)\n"
        "declare float @llvm.sqrt.f32(float)\n"
        "declare float @llvm.maxnum.f32(float, float)\n"
        "declare float @llvm.minnum.f32(float, float)\n\n"
        "attributes #0 = { nounwind }\n"
        // tanh never sets errno, so the call is pure; this lets the loop
        // vectorizer map it to a SIMD variant when a vector libm is available.
        "attributes #1 = { nounwind willreturn memory(none) }\n";
}

void emitHeader(const IRModule &m, std::ostream &os) {
  os << "// Generated by tilec. Buffers are dense row-major float32 and must not alias.\n"
        "#pragma once\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n";
  for (const Function &f : m.funcs) {
    os << "// " << f.name << ": ";
    for (int i = 0; i < f.numArgs(); ++i)
      os << f.argNames[i] << "[" << f.ops[i].shape.r << "x" << f.ops[i].shape.c << "] ";
    os << "-> out[" << f.ops[f.result].shape.r << "x" << f.ops[f.result].shape.c << "]\n";
    os << "void " << f.name << "(";
    for (int i = 0; i < f.numArgs(); ++i)
      os << "const float *" << f.argNames[i] << ", ";
    os << "float *out);\n\n";
  }
  os << "#ifdef __cplusplus\n}\n#endif\n";
}

} // namespace tilec
