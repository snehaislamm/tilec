#pragma once
#include "tilec/Diagnostics.h"
#include <string>
#include <vector>

namespace tilec {

enum class Tok {
  Eof,
  Ident,
  Number,
  KwKernel,
  KwReturn,
  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  Comma,
  Colon,
  Semi,
  Assign,
  Plus,
  Minus,
  Star,
  Slash,
  At,
  Arrow,
};

struct Token {
  Tok kind;
  std::string text;
  double num = 0.0;
  SourceLoc loc;
};

const char *tokName(Tok t);

/// Converts DSL source into tokens. '#' starts a line comment.
std::vector<Token> lex(const std::string &src);

} // namespace tilec
