#!/bin/bash
# Post-train the action pathway only -- see post_train/README.md.
#
#   bash scripts/post_train_action.sh --data <lmdb dir>
set -e
# Resolve the interpreter rather than assuming `python` is on PATH: on the
# machine this was built on it is not, and the failure is a bare
# "exec: python: not found" with nothing pointing at the cause.
PY=${PYTHON:-$(command -v python3 || command -v python)}
cd "$(dirname "$0")/.."
exec "$PY" post_train/train.py --recipe action --gpus 2 --trainchk "$@"
