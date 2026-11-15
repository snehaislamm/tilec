#pragma once
#include "tilec/Diagnostics.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tilec {

struct Expr {
  enum Kind { Num, Var, Binary, Neg, Call } kind;
  SourceLoc loc;
  double num = 0.0;                        // Num
  std::string name;                        // Var / Call callee
  char op = 0;                             // Binary: + - * / @
  std::vector<std::unique_ptr<Expr>> args; // Binary(lhs, rhs), Neg(x), Call
  int axis = -1;                           // Call: axis= keyword, -1 if absent
};

struct Stmt {
  enum Kind { Assign, Return } kind;
  SourceLoc loc;
  std::string target; // Assign
  std::unique_ptr<Expr> value;
};

struct TypeAnn {
  std::vector<int64_t> dims; // rank 0..2
  SourceLoc loc;
};

struct Param {
  std::string name;
  TypeAnn type;
  SourceLoc loc;
};

struct Kernel {
  std::string name;
  SourceLoc loc;
  std::vector<Param> params;
  TypeAnn ret;
  std::vector<Stmt> body;
  std::vector<int64_t> tile; // from @tile(m, n, k); empty if absent
};

struct Module {
  std::vector<Kernel> kernels;
};

} // namespace tilec
