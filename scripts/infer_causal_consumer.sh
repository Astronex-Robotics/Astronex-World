#!/bin/bash
# Causal generation on a consumer card (32 GB RTX 5090, or a 24 GB card with
# little else on it). Same output as scripts/infer_causal.sh -- see
# inference/causal_consumer.yaml for the measured peaks.
#
# Add --vram-limit-gb 32 to make a larger card behave like a 5090.
set -e
PY=${PYTHON:-$(command -v python3 || command -v python)}
cd "$(dirname "$0")/.."
exec "$PY" inference/generate.py --mode causal \
  --config inference/causal_consumer.yaml \
  --prompt "A wooden sailing ship on a stormy sea at dusk, rain lashing the deck." \
  --image media/examples/case_203.jpg \
  --trajectory 'w*23' \
  --out outputs/causal_consumer "$@"
