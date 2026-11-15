#!/usr/bin/env python3
"""Minimal FileCheck clone (subset of LLVM's FileCheck).

Supported directives (prefix configurable with --check-prefix):
  CHECK:        match in order, anywhere after the previous match
  CHECK-NEXT:   match on the line right after the previous match
  CHECK-SAME:   match on the same line as the previous match
  CHECK-NOT:    must not appear between the surrounding matches
  CHECK-LABEL:  like CHECK, also splits the input into independent blocks
  CHECK-COUNT-<n>: match n consecutive occurrences in order
Patterns are literal text; {{regex}} embeds a Python regular expression.
Leading/trailing whitespace is ignored and runs of spaces match flexibly.

usage: filecheck.py <check-file> [--check-prefix P] < input
"""
import argparse
import re
import sys


def compile_pattern(pat: str) -> re.Pattern:
    out, i = [], 0
    while i < len(pat):
        j = pat.find("{{", i)
        if j < 0:
            out.append(re.escape(pat[i:]))
            break
        out.append(re.escape(pat[i:j]))
        k = pat.find("}}", j)
        if k < 0:
            raise SystemExit(f"filecheck: unterminated '{{{{' in pattern: {pat}")
        out.append("(?:" + pat[j + 2:k] + ")")
        i = k + 2
    rx = "".join(out)
    rx = re.sub(r"(\\ )+", r"\\s+", rx)
    return re.compile(rx)


def parse_checks(path, prefix):
    rx = re.compile(r"\b" + re.escape(prefix) + r"(-NEXT|-SAME|-NOT|-LABEL|-COUNT-\d+)?:(.*)$")
    checks = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            m = rx.search(line)
            if m:
                kind = (m.group(1) or "").lstrip("-") or "CHECK"
                checks.append((kind, m.group(2).strip(), lineno))
    if not checks:
        raise SystemExit(f"filecheck: no {prefix}: directives found in {path}")
    return checks


def fail(path, lineno, msg, lines, near):
    sys.stderr.write(f"{path}:{lineno}: error: {msg}\n")
    lo = max(0, near - 2)
    for k in range(lo, min(len(lines), near + 3)):
        sys.stderr.write(f"  input:{k + 1}: {lines[k]}\n")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkfile")
    ap.add_argument("--check-prefix", default="CHECK")
    args = ap.parse_args()
    checks = parse_checks(args.checkfile, args.check_prefix)
    lines = sys.stdin.read().splitlines()

    line, col = 0, 0          # position just after the previous match
    pending_not = []
    prev_line = -1

    def search(pat, start_line, start_col, end_line=None):
        rx = compile_pattern(pat)
        stop = len(lines) if end_line is None else end_line
        for ln in range(start_line, stop):
            text = lines[ln]
            m = rx.search(text, start_col if ln == start_line else 0)
            if m:
                return ln, m.end()
        return None

    def check_nots(upto_line, upto_col):
        for pat, lno in pending_not:
            rx = compile_pattern(pat)
            for ln in range(line, min(upto_line + 1, len(lines))):
                text = lines[ln]
                s = col if ln == line else 0
                e = upto_col if ln == upto_line else len(text)
                if rx.search(text[s:e]):
                    fail(args.checkfile, lno, f"CHECK-NOT: pattern '{pat}' found", lines, ln)
        pending_not.clear()

    for kind, pat, lno in checks:
        if kind == "NOT":
            pending_not.append((pat, lno))
            continue
        if kind.startswith("COUNT-"):
            n = int(kind.split("-")[1])
            for _ in range(n):
                r = search(pat, line, col)
                if r is None:
                    fail(args.checkfile, lno, f"CHECK-COUNT: expected {n} matches of '{pat}'",
                         lines, line)
                check_nots(*r)
                prev_line, (line, col) = r[0], r
            continue
        if kind == "NEXT":
            target = prev_line + 1
            r = search(pat, target, 0, target + 1) if target < len(lines) else None
            if r is None:
                fail(args.checkfile, lno, f"CHECK-NEXT: expected '{pat}' on next line",
                     lines, target)
        elif kind == "SAME":
            r = search(pat, prev_line, col, prev_line + 1)
            if r is None:
                fail(args.checkfile, lno, f"CHECK-SAME: expected '{pat}' on same line",
                     lines, prev_line)
        else:
            r = search(pat, line, col)
            if r is None:
                fail(args.checkfile, lno, f"{kind}: expected string not found: '{pat}'",
                     lines, line)
        check_nots(*r)
        prev_line, (line, col) = r[0], r
    if pending_not:
        check_nots(len(lines) - 1, len(lines[-1]) if lines else 0)


if __name__ == "__main__":
    main()
