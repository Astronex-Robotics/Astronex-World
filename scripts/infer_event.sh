#!/bin/bash
# An event injected partway through: the scene is established first, then the
# caption changes and the event lands mid-video.
#
# Two things about the event text are not obvious and both were learned the
# hard way. It is *appended* to the caption, never substituted -- the release
# has no dedicated event branch, so the event reaches the model through the
# same caption cross-attention as everything else. And it has to stay in plain
# language: rewriting the same instruction into technical vocabulary stops it
# triggering at all.
set -e
# Resolve the interpreter rather than assuming `python` is on PATH: on the
# machine this was built on it is not, and the failure is a bare
# "exec: python: not found" with nothing pointing at the cause.
PY=${PYTHON:-$(command -v python3 || command -v python)}
cd "$(dirname "$0")/.."
exec "$PY" inference/generate.py --mode causal \
  --prompt "A grand magical library interior in CG style, tall shelves and floating candles, a wizard standing at a reading desk." \
  --event-prompt "The wizard picks up the crystal staff." \
  --event-start-frame 8 \
  --trajectory 'h*23' \
  --out outputs/causal_event "$@"
