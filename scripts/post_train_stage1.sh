#!/bin/bash
# Stage "stage1" -- see post_train/README.md for what it trains and why.
set -e
# Resolve the interpreter rather than assuming `python` is on PATH: on the
# machine this was built on it is not, and the failure is a bare
# "exec: python: not found" with nothing pointing at the cause.
PY=${PYTHON:-$(command -v python3 || command -v python)}
cd "$(dirname "$0")/.."
exec "$PY" post_train/train.py --recipe stage1 --gpus 2 --trainchk "$@"
