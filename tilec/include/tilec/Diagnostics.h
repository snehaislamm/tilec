#pragma once
#include <stdexcept>
#include <string>

namespace tilec {

struct SourceLoc {
  int line = 0;
  int col = 0;
};

/// Thrown for any user-facing compile error (lex, parse, type/shape).
class CompileError : public std::runtime_error {
public:
  CompileError(SourceLoc loc, const std::string &msg) : std::runtime_error(msg), loc(loc) {}
  SourceLoc loc;
};

} // namespace tilec
