"""Post-train Astronex-World.

Two recipes, each a copy of a config that actually ran here, with the resume
fields reset so it starts from the release rather than mid-run. Which control
pathways train is a property of the stage, not a free choice -- that is what
`train_control_params` in each config says.

| recipe | trains                |
|--------|-----------------------|
| camera | camera (PRoPE)        |
| action | action in, action out |
| sft    | camera, action        |

**camera** and **action** each train one pathway and leave the other where the
release left it, which is what makes "did this corpus change camera control?"
answerable. **sft** trains both together: camera + action refinement on a new
corpus, 36.1 px of ball bounce against the teacher's 88.8.

The action head is supervised by `_action_output_loss` in
`post_train/objectives/camera_diffusion.py`, and only `action` marks it
trainable. The recipes that do not train it also do not install it: a frozen
zero-initialised head returns zeros and has no gradient to give, so charging
its loss would cost a forward pass and teach nothing.

Both keep the backbone frozen and train LoRA adapters plus the named control
grafts, so the released picture quality is the starting point, not something
these recipes retrain.

NCCL is the backend. Single node is the default; `--nnodes`/`--node-rank`/
`--master-addr` extend it across hosts.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import astronex_env  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

RECIPES = {
    "camera":  ("camera.yaml",
                "camera (PRoPE) only; the action pathway is left alone"),
    "action":  ("action.yaml",
                "action in and out only; the camera pathway is left alone"),
    "sft":     ("sft.yaml",
                "camera + action together"),
}


def main():
    ap = argparse.ArgumentParser(
        description="Astronex-World post-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="recipes:\n" + "\n".join(
            f"  {k:8s} {d}" for k, (_, d) in RECIPES.items()))
    ap.add_argument("--recipe", choices=list(RECIPES), default="sft")
    ap.add_argument("--config", default=None,
                    help="explicit config path, overriding --recipe")
    ap.add_argument("--logdir", default=None, help="overrides the config")
    ap.add_argument("--weights", default=None,
                    help="checkpoint the \"@weights\" entries resolve to "
                         "(default: the released directory)")
    ap.add_argument("--data", default=None,
                    help="training data, overriding the config's data_path")
    ap.add_argument("--gpus", type=int, default=2, help="processes on this node")
    ap.add_argument("--nnodes", type=int, default=1)
    ap.add_argument("--node-rank", type=int, default=0)
    ap.add_argument("--master-addr", default="localhost")
    ap.add_argument("--port", type=int, default=30401)
    ap.add_argument("--sp-size", type=int, default=1,
                    help="sequence-parallel group size; >1 switches to the "
                         "SP-aware init and shards the sequence across ranks")
    ap.add_argument("--memlog", action="store_true",
                    help="resident and peak memory per phase")
    ap.add_argument("--trainchk", action="store_true",
                    help="per step, the generator's gradient norm, how far the "
                         "optimiser moved it, and which parameter groups got "
                         "no gradient at all. A norm of 0 with every grad None "
                         "means the run computes nothing and its checkpoints "
                         "will come out identical to their starting point")
    ap.add_argument("--nccl-debug", action="store_true",
                    help="NCCL_DEBUG=INFO, for a multi-node job that hangs on "
                         "its first collective instead of failing")
    ap.add_argument("--wandb", action="store_true")
    args, passthrough = ap.parse_known_args()

    config = (os.path.abspath(args.config) if args.config else
              os.path.join(HERE, "configs", RECIPES[args.recipe][0]))
    if not os.path.exists(config):
        raise SystemExit(f"config not found: {config}")
    weights = os.path.abspath(args.weights) if args.weights else None
    data = os.path.abspath(args.data) if args.data else None
    logdir = os.path.abspath(args.logdir) if args.logdir else None
    root = astronex_env.setup()
    _check_scheme(config, args.recipe)
    from omegaconf import OmegaConf
    logdir = logdir or os.path.join(
        root, str(OmegaConf.load(config).get("logdir", "logs/astronex_post")))
    config = astronex_env.resolve_config(
        config, os.path.join(logdir, "_astronex_config.yaml"),
        weights_dir=weights, data=data)

    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nnodes={args.nnodes}",
           f"--node_rank={args.node_rank}",
           f"--master_addr={args.master_addr}",
           f"--master_port={args.port}",
           f"--nproc_per_node={args.gpus}",
           "-m", "post_train.run_train",
           "--config_path", config,
           "--sp_size", str(args.sp_size),
           "--tf", "--allow-small-bsz"]
    cmd += ["--logdir", logdir]
    cmd += passthrough

    env = astronex_env.child_env(weights or astronex_env.weights())
    if not args.wandb:
        env.setdefault("WANDB_MODE", "disabled")
    if args.memlog:
        env["ASTRONEX_MEMLOG"] = "1"
    if args.trainchk:
        env["ASTRONEX_TRAINCHK"] = "1"
    if args.nccl_debug:
        env["NCCL_DEBUG"] = "INFO"
    if args.nnodes > 1:
        # Single-node NCCL finds its transport on its own; across hosts it
        # picks an interface, and on a box with several it can pick one the
        # peers cannot reach. Name it rather than letting it guess.
        env.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")
        env.setdefault("NCCL_IB_DISABLE", "0")

    if args.config:
        print(f"[astronex] explicit config (recipe ignored)")
    else:
        print(f"[astronex] recipe {args.recipe}: {RECIPES[args.recipe][1]}")
    print(f"[astronex] config {config}")
    print(f"[astronex] {args.nnodes} node(s) x {args.gpus} gpu(s), "
          f"sp_size {args.sp_size}, backend nccl")
    print("[astronex] " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=root, env=env))


def _check_scheme(config, recipe):
    """Report which pathways a config trains, and refuse a combination that
    reads as enabled and is not."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config)
    lora = cfg.model_kwargs.get("lora", None)
    tcp = lora.get("train_control_params", False) if lora else False
    names = ([] if tcp in (False, None) else
             (["camera", "action", "action_out", "event", "context"]
              if tcp is True else list(tcp)))
    trainer = str(cfg.get("trainer", "?"))
    if trainer != "camera_diffusion":
        raise SystemExit(f"trainer {trainer!r} is not part of this release; "
                         f"expected camera_diffusion")

    print(f"[astronex] trainer {trainer}")
    print("[astronex] control pathways: "
          f"{', '.join(names) or 'none (adapters only)'}")

    # `split_dtype_fsdp` leaves trainable parameters fp32 and casts the frozen
    # backbone to bf16. LoRA survives that because `LoRALinear.forward` casts
    # at its own boundary; the grafts are plain `nn.Linear` and do not, so an
    # fp32 `prope_o` meets a bf16 activation and raises minutes in. Casting the
    # grafts down instead is worse -- at lr 2e-6 to 2e-5 a bf16 master rounds
    # the updates to nothing, which trains the pathway not at all and says so
    # nowhere.
    if names and bool(cfg.get("split_dtype_fsdp", False)):
        raise SystemExit(
            "split_dtype_fsdp cannot be combined with trainable grafts: they "
            "have no dtype boundary of their own, and casting them to bf16 "
            "would round lr-sized updates away. Set it false in this config.")

    if "event" in names:
        # Zero-initialised at `o`, so it begins as the identity and becomes
        # something only if the data carries event text.
        print("[astronex] event branch on: rows without event text get the "
              "empty-string embedding, so every rank runs the same modules -- "
              "a rank that skipped it would hang for the whole NCCL timeout "
              "rather than raise")
    if {"action", "action_out"} & set(names):
        print("[astronex] note: 12 of the 16 shards in dataset/sft_phys carry "
              "all-zero actions, and control2v has no action column at all. "
              "Only crossfps (3700 rows) and dojo (44) hold real action "
              "signal; on the rest the loss reads ~0 and teaches nothing.")


if __name__ == "__main__":
    main()
