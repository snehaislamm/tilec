#!/usr/bin/env python3
"""`not <cmd>`: succeed iff cmd exits with 1 or 2 (a diagnosed error).

Exit codes >= 3 (internal compiler error) and signals (crashes) still fail, so
negative tests cannot accidentally pass on an ICE.
"""
import subprocess
import sys

r = subprocess.run(sys.argv[1:])
sys.exit(0 if r.returncode in (1, 2) else 1)
