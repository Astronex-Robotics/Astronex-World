#!/bin/bash
# Causal: 8 steps, sliding 20-frame window, 4-frame sink, retrieval on.
# Streams block by block, so the rollout can be extended and driven.
set -e
# Resolve the interpreter rather than assuming `python` is on PATH: on the
# machine this was built on it is not, and the failure is a bare
# "exec: python: not found" with nothing pointing at the cause.
PY=${PYTHON:-$(command -v python3 || command -v python)}
cd "$(dirname "$0")/.."
exec "$PY" inference/generate.py --mode causal \
  --prompt "A wooden sailing ship on a stormy sea at dusk, rain lashing the deck." \
  --image ../minWM/data/wbench/images/case_203.jpg \
  --trajectory 'w*23' \
  --out outputs/causal_pirate "$@"
