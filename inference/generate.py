"""Generate video with Astronex-World, causal or bidirectional.

Both modes run the same 5.35B backbone and differ in two things that have to
move together:

* `causal` attends only backwards over a sliding 20-frame window with a
  4-frame sink, and samples in 8 steps. It generates block by block, so a
  rollout can be extended indefinitely and a camera or action stream can be
  fed to it as it goes. This is the distilled student.
* `bidirectional` attends over the whole clip at once and samples in 50 steps.
  It cannot stream and cannot be extended past its window, but it keeps far
  more of the physics -- measured on a dropped ball, 88.8 px of bounce against
  the student's 16.6. This is the teacher the student was distilled from.

Pick causal for interaction and length, bidirectional for a fixed-length clip
where motion has to be right.

This is a thin front end: it resolves the released weights and the mode's
config and hands both to `inference/sample.py`, which is where the sampling
loop, the retrieval window and the decoding live.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import astronex_env  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(
        description="Astronex-World generation (causal or bidirectional)")
    ap.add_argument("--mode", choices=("causal", "bidirectional"),
                    default="causal")
    ap.add_argument("--prompt", required=True,
                    help="prompt text, or a path to a .txt of prompts")
    ap.add_argument("--image", default=None,
                    help="reference image; omit for text-to-video")
    ap.add_argument("--trajectory", default=None,
                    help="camera stream, e.g. 'w*23' forward, 'h*23' hold, "
                         "'a*12,d*11' left then right. Defaults to holding "
                         "still for the whole clip; its length has to equal "
                         "the frame count")
    ap.add_argument("--frames", type=int, default=None,
                    help="latent frames to generate (default: 23 causal, "
                         "17 bidirectional)")
    ap.add_argument("--out", default="outputs/astronex")
    ap.add_argument("--steps", type=int, default=None,
                    help="override the sampler steps. 8 causal / 50 "
                         "bidirectional are the validated values; 4 causal "
                         "degrades visibly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--config", default=None,
                    help="sampling config; defaults to inference/<mode>.yaml. "
                         "inference/causal_consumer.yaml fits a 32 GB card")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--vram-limit-gb", type=float, default=None,
                    help="cap this process's VRAM, e.g. 32 to behave like an "
                         "RTX 5090 on a larger card")
    ap.add_argument("--weights", default=None,
                    help="weights directory to sample from. Defaults to the "
                         "released directory (../Astronex, or "
                         "ASTRONEX_WEIGHTS), shared by both modes")
    ap.add_argument("--event-prompt", default=None,
                    help="second caption, swapped in mid-rollout")
    ap.add_argument("--event-start-frame", type=int, default=None)
    args, passthrough = ap.parse_known_args()
    # Resolve paths against the caller's directory before setup() chdirs.
    args.out = os.path.abspath(args.out)
    if args.config:
        args.config = os.path.abspath(args.config)
    if args.image:
        args.image = os.path.abspath(args.image)
    if os.path.exists(args.prompt):
        args.prompt = os.path.abspath(args.prompt)

    root = astronex_env.setup()
    weights = os.path.abspath(args.weights) if args.weights else \
        astronex_env.weights(args.mode)
    out_dir = args.out
    config = astronex_env.resolve_config(
        args.config or os.path.join(HERE, f"{args.mode}.yaml"),
        os.path.join(out_dir, f"_astronex_{args.mode}.yaml"), weights_dir=weights)
    frames = args.frames if args.frames is not None else (
        23 if args.mode == "causal" else 17)

    # The causal pipeline generates whole blocks, and i2v spends one frame on
    # the reference, so `1 + frames` has to divide by the block size -- 23, not
    # 20, at block 8. Snapping is friendlier than the ValueError three minutes
    # into model loading.
    if args.mode == "causal":
        block = 8
        need = (block - (1 + frames) % block) % block if args.image else \
               (block - frames % block) % block
        if need:
            print(f"[astronex] frames {frames} -> {frames + need}: the causal "
                  f"pipeline needs complete blocks of {block}"
                  + (" including the i2v reference frame" if args.image else ""))
            frames += need

    trajectory = args.trajectory or f"h*{frames}"

    # Refuse this before a 20 GB load rather than after it: the bidirectional
    # pipeline conditions the whole rollout in one pass, so there is no "before
    # the event" to keep identical, and it raises from deep inside inference.
    if args.event_start_frame and args.mode != "causal":
        raise SystemExit(
            "--event-start-frame is only supported by the causal pipeline. "
            "The bidirectional pipeline conditions the whole clip at once; "
            "drop the flag, or pass --event-prompt alone to append it as a "
            "static caption.")

    # A prompt on the command line still has to reach the loader as a file;
    # the dataset it reads is line-oriented. Write it into this run's output
    # directory so two concurrent runs do not overwrite each other's prompt.
    prompt_path = args.prompt
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(out_dir, "_astronex_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(args.prompt.rstrip("\n") + "\n")

    cmd = [sys.executable, "-m", "inference.sample",
           "--config_path", config,
           "--checkpoint_path", weights,
           "--data_path", prompt_path,
           "--output_folder", out_dir,
           "--num_output_frames", str(frames),
           "--trajectory", trajectory,
           "--seed", str(args.seed),
           "--fps", str(args.fps)]
    if args.steps is not None:
        cmd += ["--sampling_steps", str(args.steps)]
    if args.image:
        cmd += ["--i2v", "--image_path", args.image]
    if args.event_prompt:
        cmd += ["--event_prompt", args.event_prompt]
    if args.event_start_frame is not None:
        cmd += ["--event_start_frame", str(args.event_start_frame)]
    cmd += passthrough

    env = astronex_env.child_env(weights)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.vram_limit_gb:
        env["ASTRONEX_VRAM_LIMIT_GB"] = str(args.vram_limit_gb)

    print(f"[astronex] {args.mode} | weights {weights}")
    print("[astronex] " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=root, env=env))


if __name__ == "__main__":
    main()
