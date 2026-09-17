#!/bin/bash
# Text-to-video: no reference image, same weights and same sampler as the i2v
# scripts. The camera stream still applies, and its length has to match the
# frame count -- 24 here, because without an i2v reference frame the causal
# pipeline wants whole blocks of 8.
set -e
# Resolve the interpreter rather than assuming `python` is on PATH: on the
# machine this was built on it is not, and the failure is a bare
# "exec: python: not found" with nothing pointing at the cause.
PY=${PYTHON:-$(command -v python3 || command -v python)}
cd "$(dirname "$0")/.."
exec "$PY" inference/generate.py --mode causal \
  --prompt "A wooden sailing ship on a stormy sea at dusk, rain lashing the deck." \
  --trajectory 'w*24' \
  --out outputs/t2v_ship "$@"
