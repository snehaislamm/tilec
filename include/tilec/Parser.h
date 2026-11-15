#pragma once
#include "tilec/AST.h"
#include "tilec/Lexer.h"

namespace tilec {

/// Recursive-descent parser.
///
///   module   := (attr* kernel)*
///   attr     := '@' ident '(' number (',' number)* ')'
///   kernel   := 'kernel' ident '(' params? ')' '->' type '{' stmt* '}'
///   param    := ident ':' type
///   type     := 'f32' ('[' int (',' int)* ']')?
///   stmt     := ident '=' expr ';'? | 'return' expr ';'?
///   expr     := term (('+' | '-') term)*
///   term     := unary (('*' | '/' | '@') unary)*
///   unary    := '-' unary | primary
///   primary  := number | ident | ident '(' args ')' | '(' expr ')'
///   args     := arg (',' arg)*      arg := expr | 'axis' '=' int
Module parse(const std::vector<Token> &toks);

} // namespace tilec
