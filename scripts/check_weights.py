"""Check that the released weights directory has everything Astronex loads.

    python scripts/check_weights.py            # ../Astronex, or ASTRONEX_WEIGHTS

A missing text encoder otherwise surfaces minutes into a run, as a tokenizer
error after the denoiser has already loaded.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import astronex_env  # noqa: E402


def main():
    root = astronex_env.WEIGHTS_ROOT
    print(f"weights {root}")
    ok = os.path.isdir(root)
    for c in astronex_env.COMPONENTS:
        here = os.path.exists(os.path.join(root, c))
        ok &= here
        print(f"  {'ok  ' if here else 'MISS'}  {c}")
    index = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(index):
        shards = sorted(set(json.load(open(index))["weight_map"].values()))
        present = [s for s in shards if os.path.exists(os.path.join(root, s))]
        print(f"  {'ok  ' if len(present) == len(shards) else 'MISS'}  "
              f"denoiser shards {len(present)}/{len(shards)}")
        ok &= len(present) == len(shards)
    else:
        print("  MISS  model.safetensors.index.json")
        ok = False
    print("\nall present" if ok else
          "\nsomething is missing: download Astronex-Lab/Astronex-World "
          "or point ASTRONEX_WEIGHTS at it")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
