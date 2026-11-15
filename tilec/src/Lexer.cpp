#include "tilec/Lexer.h"
#include <cctype>
#include <cstdlib>

namespace tilec {

const char *tokName(Tok t) {
  switch (t) {
  case Tok::Eof: return "end of file";
  case Tok::Ident: return "identifier";
  case Tok::Number: return "number";
  case Tok::KwKernel: return "'kernel'";
  case Tok::KwReturn: return "'return'";
  case Tok::LParen: return "'('";
  case Tok::RParen: return "')'";
  case Tok::LBrace: return "'{'";
  case Tok::RBrace: return "'}'";
  case Tok::LBracket: return "'['";
  case Tok::RBracket: return "']'";
  case Tok::Comma: return "','";
  case Tok::Colon: return "':'";
  case Tok::Semi: return "';'";
  case Tok::Assign: return "'='";
  case Tok::Plus: return "'+'";
  case Tok::Minus: return "'-'";
  case Tok::Star: return "'*'";
  case Tok::Slash: return "'/'";
  case Tok::At: return "'@'";
  case Tok::Arrow: return "'->'";
  }
  return "?";
}

std::vector<Token> lex(const std::string &src) {
  std::vector<Token> out;
  size_t i = 0;
  int line = 1, col = 1;
  auto adv = [&]() {
    if (src[i] == '\n') {
      ++line;
      col = 1;
    } else {
      ++col;
    }
    ++i;
  };
  while (i < src.size()) {
    char c = src[i];
    if (std::isspace(static_cast<unsigned char>(c))) {
      adv();
      continue;
    }
    if (c == '#') {
      while (i < src.size() && src[i] != '\n')
        adv();
      continue;
    }
    SourceLoc loc{line, col};
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      std::string s;
      while (i < src.size() &&
             (std::isalnum(static_cast<unsigned char>(src[i])) || src[i] == '_')) {
        s += src[i];
        adv();
      }
      Tok k = Tok::Ident;
      if (s == "kernel")
        k = Tok::KwKernel;
      else if (s == "return")
        k = Tok::KwReturn;
      out.push_back({k, s, 0.0, loc});
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) ||
        (c == '.' && i + 1 < src.size() && std::isdigit(static_cast<unsigned char>(src[i + 1])))) {
      std::string s;
      bool seenExp = false;
      while (i < src.size()) {
        char d = src[i];
        if (std::isdigit(static_cast<unsigned char>(d)) || d == '.') {
          s += d;
          adv();
        } else if ((d == 'e' || d == 'E') && !seenExp) {
          seenExp = true;
          s += d;
          adv();
          if (i < src.size() && (src[i] == '+' || src[i] == '-')) {
            s += src[i];
            adv();
          }
        } else {
          break;
        }
      }
      char *end = nullptr;
      double v = std::strtod(s.c_str(), &end);
      if (!end || *end != '\0') throw CompileError(loc, "malformed number literal '" + s + "'");
      out.push_back({Tok::Number, s, v, loc});
      continue;
    }
    Tok k;
    switch (c) {
    case '(': k = Tok::LParen; break;
    case ')': k = Tok::RParen; break;
    case '{': k = Tok::LBrace; break;
    case '}': k = Tok::RBrace; break;
    case '[': k = Tok::LBracket; break;
    case ']': k = Tok::RBracket; break;
    case ',': k = Tok::Comma; break;
    case ':': k = Tok::Colon; break;
    case ';': k = Tok::Semi; break;
    case '=': k = Tok::Assign; break;
    case '+': k = Tok::Plus; break;
    case '*': k = Tok::Star; break;
    case '/': k = Tok::Slash; break;
    case '@': k = Tok::At; break;
    case '-':
      if (i + 1 < src.size() && src[i + 1] == '>') {
        adv();
        adv();
        out.push_back({Tok::Arrow, "->", 0.0, loc});
        continue;
      }
      k = Tok::Minus;
      break;
    default: throw CompileError(loc, std::string("unexpected character '") + c + "'");
    }
    out.push_back({k, std::string(1, c), 0.0, loc});
    adv();
  }
  out.push_back({Tok::Eof, "", 0.0, {line, col}});
  return out;
}

} // namespace tilec
