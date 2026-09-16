#!/bin/bash
# Bidirectional: 50 steps over the whole 17-frame window at once. No streaming
# and no extending, but it keeps the physics the causal student loses.
set -e
# Resolve the interpreter rather than assuming `python` is on PATH: on the
# machine this was built on it is not, and the failure is a bare
# "exec: python: not found" with nothing pointing at the cause.
PY=${PYTHON:-$(command -v python3 || command -v python)}
cd "$(dirname "$0")/.."
exec "$PY" inference/generate.py --mode bidirectional \
  --prompt "A wooden sailing ship on a stormy sea at dusk, rain lashing the deck." \
  --image ../minWM/data/wbench/images/case_203.jpg \
  --out outputs/bidir_pirate "$@"
