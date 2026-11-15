#include "tilec/Parser.h"
#include <cmath>

namespace tilec {
namespace {

class Parser {
public:
  explicit Parser(const std::vector<Token> &t) : toks(t) {}

  Module parseModule() {
    Module m;
    while (!at(Tok::Eof)) {
      std::vector<int64_t> tile;
      while (at(Tok::At)) {
        SourceLoc loc = cur().loc;
        next();
        Token name = expect(Tok::Ident, "attribute name");
        if (name.text != "tile")
          throw CompileError(name.loc, "unknown attribute '@" + name.text + "'");
        expect(Tok::LParen, "'('");
        tile.push_back(parseInt("tile size"));
        while (accept(Tok::Comma))
          tile.push_back(parseInt("tile size"));
        expect(Tok::RParen, "')'");
        if (tile.size() != 3) throw CompileError(loc, "@tile expects exactly 3 sizes (m, n, k)");
        for (int64_t v : tile)
          if (v <= 0) throw CompileError(loc, "@tile sizes must be positive");
      }
      Kernel k = parseKernel();
      k.tile = tile;
      m.kernels.push_back(std::move(k));
    }
    return m;
  }

private:
  const std::vector<Token> &toks;
  size_t pos = 0;

  const Token &cur() const { return toks[pos]; }
  const Token &peek(size_t n = 1) const { return toks[std::min(pos + n, toks.size() - 1)]; }
  bool at(Tok k) const { return cur().kind == k; }
  void next() {
    if (pos + 1 < toks.size()) ++pos;
  }
  bool accept(Tok k) {
    if (at(k)) {
      next();
      return true;
    }
    return false;
  }
  Token expect(Tok k, const char *what) {
    if (!at(k))
      throw CompileError(cur().loc, std::string("expected ") + what + ", found " +
                                        (cur().kind == Tok::Eof ? std::string("end of file")
                                                                : "'" + cur().text + "'"));
    Token t = cur();
    next();
    return t;
  }
  int64_t parseInt(const char *what) {
    Token t = expect(Tok::Number, what);
    if (t.num != std::floor(t.num) || t.num < 0)
      throw CompileError(t.loc, std::string(what) + " must be a non-negative integer");
    return static_cast<int64_t>(t.num);
  }

  TypeAnn parseType() {
    TypeAnn ty;
    Token t = expect(Tok::Ident, "type");
    ty.loc = t.loc;
    if (t.text != "f32")
      throw CompileError(t.loc,
                         "unsupported element type '" + t.text + "' (only f32 is supported)");
    if (accept(Tok::LBracket)) {
      ty.dims.push_back(parseInt("dimension"));
      while (accept(Tok::Comma))
        ty.dims.push_back(parseInt("dimension"));
      expect(Tok::RBracket, "']'");
      if (ty.dims.size() > 2) throw CompileError(t.loc, "tensors of rank > 2 are not supported");
      for (int64_t d : ty.dims)
        if (d <= 0) throw CompileError(t.loc, "dimensions must be positive");
    }
    return ty;
  }

  Kernel parseKernel() {
    Kernel k;
    k.loc = expect(Tok::KwKernel, "'kernel'").loc;
    k.name = expect(Tok::Ident, "kernel name").text;
    expect(Tok::LParen, "'('");
    if (!at(Tok::RParen)) {
      do {
        Param p;
        Token n = expect(Tok::Ident, "parameter name");
        p.name = n.text;
        p.loc = n.loc;
        expect(Tok::Colon, "':'");
        p.type = parseType();
        k.params.push_back(std::move(p));
      } while (accept(Tok::Comma));
    }
    expect(Tok::RParen, "')'");
    expect(Tok::Arrow, "'->'");
    k.ret = parseType();
    expect(Tok::LBrace, "'{'");
    while (!at(Tok::RBrace)) {
      if (at(Tok::Eof)) throw CompileError(cur().loc, "unterminated kernel body, expected '}'");
      k.body.push_back(parseStmt());
    }
    expect(Tok::RBrace, "'}'");
    return k;
  }

  Stmt parseStmt() {
    Stmt s;
    s.loc = cur().loc;
    if (accept(Tok::KwReturn)) {
      s.kind = Stmt::Return;
      s.value = parseExpr();
    } else {
      s.kind = Stmt::Assign;
      s.target = expect(Tok::Ident, "statement").text;
      expect(Tok::Assign, "'='");
      s.value = parseExpr();
    }
    accept(Tok::Semi);
    return s;
  }

  std::unique_ptr<Expr> binary(char op, SourceLoc loc, std::unique_ptr<Expr> l,
                               std::unique_ptr<Expr> r) {
    auto e = std::make_unique<Expr>();
    e->kind = Expr::Binary;
    e->op = op;
    e->loc = loc;
    e->args.push_back(std::move(l));
    e->args.push_back(std::move(r));
    return e;
  }

  std::unique_ptr<Expr> parseExpr() {
    auto lhs = parseTerm();
    while (at(Tok::Plus) || at(Tok::Minus)) {
      Token op = cur();
      next();
      lhs = binary(op.text[0], op.loc, std::move(lhs), parseTerm());
    }
    return lhs;
  }

  std::unique_ptr<Expr> parseTerm() {
    auto lhs = parseUnary();
    while (at(Tok::Star) || at(Tok::Slash) || at(Tok::At)) {
      Token op = cur();
      next();
      lhs = binary(op.text[0], op.loc, std::move(lhs), parseUnary());
    }
    return lhs;
  }

  std::unique_ptr<Expr> parseUnary() {
    if (at(Tok::Minus)) {
      SourceLoc loc = cur().loc;
      next();
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Neg;
      e->loc = loc;
      e->args.push_back(parseUnary());
      return e;
    }
    return parsePrimary();
  }

  std::unique_ptr<Expr> parsePrimary() {
    auto e = std::make_unique<Expr>();
    e->loc = cur().loc;
    if (at(Tok::Number)) {
      e->kind = Expr::Num;
      e->num = cur().num;
      next();
      return e;
    }
    if (accept(Tok::LParen)) {
      auto inner = parseExpr();
      expect(Tok::RParen, "')'");
      return inner;
    }
    Token id = expect(Tok::Ident, "expression");
    e->name = id.text;
    if (!accept(Tok::LParen)) {
      e->kind = Expr::Var;
      return e;
    }
    e->kind = Expr::Call;
    if (!at(Tok::RParen)) {
      do {
        if (at(Tok::Ident) && cur().text == "axis" && peek().kind == Tok::Assign) {
          SourceLoc aloc = cur().loc;
          next();
          next();
          if (e->axis != -1) throw CompileError(aloc, "duplicate 'axis' argument");
          e->axis = static_cast<int>(parseInt("axis"));
          if (e->axis > 1) throw CompileError(aloc, "axis must be 0 or 1");
        } else {
          e->args.push_back(parseExpr());
        }
      } while (accept(Tok::Comma));
    }
    expect(Tok::RParen, "')'");
    return e;
  }
};

} // namespace

Module parse(const std::vector<Token> &toks) {
  return Parser(toks).parseModule();
}

} // namespace tilec
