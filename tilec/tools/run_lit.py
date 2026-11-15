#!/usr/bin/env python3
"""Tiny lit-style test runner.

Each test file contains one or more `# RUN: <shell command>` lines.
Substitutions:  %s -> test file, %tilec -> compiler, %filecheck -> FileCheck
clone, %t -> per-test temp path prefix.  `not <cmd>` inverts the exit status
(a crash, exit code >= 3, still fails).  A test passes if every RUN line succeeds.
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_line(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilec", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    tests = []
    for p in args.paths:
        p = Path(p)
        tests += sorted(p.rglob("*.tl")) if p.is_dir() else [p]

    fc = f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'filecheck.py'))}"
    not_tool = f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'not.py'))}"
    failed = []
    tmp = Path(tempfile.mkdtemp(prefix="tilec_lit_"))
    for t in tests:
        runs = [l.split("RUN:", 1)[1].strip() for l in t.read_text().splitlines()
                if "RUN:" in l]
        if not runs:
            continue
        ok = True
        for r in runs:
            cmd = (r.replace("%tilec", shlex.quote(args.tilec))
                    .replace("%filecheck", fc)
                    .replace("%s", shlex.quote(str(t)))
                    .replace("%t", shlex.quote(str(tmp / t.stem))))
            if cmd.startswith("not "):
                cmd = not_tool + " " + cmd[4:]
            cmd = cmd.replace("| not ", "| " + not_tool + " ")
            res = run_line(cmd)
            if res.returncode != 0:
                ok = False
                print(f"FAIL: {t.name}\n  $ {r}\n{res.stdout}{res.stderr}")
                break
        if ok:
            if args.verbose:
                print(f"PASS: {t.name}")
        else:
            failed.append(t)
    total = len(tests)
    print(f"\n{total - len(failed)}/{total} lit tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
