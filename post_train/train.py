"""Post-train Astronex-World, following the staging this model was built with.

Four recipes, each a copy of a config that actually ran here, with the resume
fields reset so it starts from the release rather than mid-run. Which control
pathways train is a property of the stage, not a free choice -- that is what
`train_control_params` in each config says, and combinations outside these four
were never validated here. `dmd` plus the event branch, for one, does not fit
on two 44 GB cards: the branch adds 1.5B trainable parameters on top of DMD's
two resident networks.

| recipe | trainer                        | trains                    |
|--------|--------------------------------|---------------------------|
| stage0 | camera_bidirectional_diffusion | adapters + every graft    |
| stage1 | camera_diffusion               | action, event, action_out |
| sft    | camera_diffusion               | camera, action            |
| dmd    | camera_score_distillation      | adapters only             |

**stage0** is how the bidirectional teacher was made: full attention, no action
pathway (`use_action: false`), every graft trainable. It is also the only stage
that trains the camera pathway from scratch.

**stage1** grafts on the event branch and the action head, and is the only
stage that supervises the action head at all -- `_action_output_loss` lives in
`camera_diffusion.py`, so the DMD objective never touches it. A head marked
trainable under DMD gets `grad is None` every step.

**sft** is camera + action refinement, and reached the best physics of any
student here: 36.1 px of ball bounce against the teacher's 88.8, where DMD
lands at 19-24%. What it cannot give you is few-step sampling.

**dmd** is what buys eight-step sampling, at that cost in physics. It trains
the adapters only; the grafts stay wherever the earlier stages left them. Three
terms push back: a motion-preservation term against the clip's own temporal
difference (worth 10% -> 19% of teacher bounce), rolling forcing every step,
and a GAN against the residual noise -- a 14M pooled discriminator head on the
critic.

Read `gan_d_loss`, `d_real` and `d_fake` as the check that the adversarial pair
is adversarial. `d_loss` pinned at 1.386 -- 2*ln(2), chance exactly -- with
`d_real` and `d_fake` equal means the discriminator is learning nothing and the
generator's term is spending two forward passes pushing against it.

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
    "stage0": ("stage0_bidir.yaml",
               "bidirectional teacher: adapters + every graft, no action path"),
    "stage1": ("stage1_event_action.yaml",
               "graft on the event branch and the action head"),
    "sft":    ("sft.yaml",
               "camera + action refinement; best physics, no few-step sampling"),
    "dmd":    ("dmd.yaml",
               "distil to eight steps; adapters only, grafts frozen"),
}


def main():
    ap = argparse.ArgumentParser(
        description="Astronex-World post-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="recipes:\n" + "\n".join(
            f"  {k:8s} {d}" for k, (_, d) in RECIPES.items()))
    ap.add_argument("--recipe", choices=list(RECIPES), default="dmd")
    ap.add_argument("--config", default=None,
                    help="explicit config path, overriding --recipe")
    ap.add_argument("--logdir", default=None, help="overrides the config")
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

    root = astronex_env.setup()
    config = (os.path.abspath(args.config) if args.config else
              os.path.join(HERE, "configs", RECIPES[args.recipe][0]))
    if not os.path.exists(config):
        raise SystemExit(f"config not found: {config}")
    _check_scheme(config, args.recipe)

    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nnodes={args.nnodes}",
           f"--node_rank={args.node_rank}",
           f"--master_addr={args.master_addr}",
           f"--master_port={args.port}",
           f"--nproc_per_node={args.gpus}",
           "Wan21/wan_train.py",
           "--config_path", config,
           "--sp_size", str(args.sp_size),
           "--tf", "--allow-small-bsz"]
    if args.logdir:
        cmd += ["--logdir", args.logdir]
    cmd += passthrough

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(root, "Wan21"), os.path.join(root, "shared"),
         env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # Fragmentation, not capacity, is what this run kept dying on: 84 MiB short
    # with 745 MiB reserved and unallocated.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not args.wandb:
        env.setdefault("WANDB_MODE", "disabled")
    if args.memlog:
        env["MINWM_MEMLOG"] = "1"
    if args.trainchk:
        env["MINWM_TRAINCHK"] = "1"
    if args.nccl_debug:
        env["NCCL_DEBUG"] = "INFO"
    if args.nnodes > 1:
        # Single-node NCCL finds its transport on its own; across hosts it
        # picks an interface, and on a box with several it can pick one the
        # peers cannot reach. Name it rather than letting it guess.
        env.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")
        env.setdefault("NCCL_IB_DISABLE", "0")

    print(f"[astronex] recipe {args.recipe}: {RECIPES[args.recipe][1]}")
    print(f"[astronex] config {config}")
    print(f"[astronex] {args.nnodes} node(s) x {args.gpus} gpu(s), "
          f"sp_size {args.sp_size}, backend nccl")
    print("[astronex] " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=root, env=env))


def _check_scheme(config, recipe):
    """Report which pathways a config trains, and refuse two combinations.

    Both refusals are for settings that read as enabled and are not.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config)
    lora = cfg.model_kwargs.get("lora", None)
    tcp = lora.get("train_control_params", False) if lora else False
    names = ([] if tcp in (False, None) else
             (["camera", "action", "action_out", "event", "context"]
              if tcp is True else list(tcp)))
    trainer = str(cfg.get("trainer", "?"))

    print(f"[astronex] trainer {trainer}")
    print("[astronex] control pathways: "
          f"{', '.join(names) or 'none (adapters only)'}")

    # The action-prediction loss (`_action_output_loss`, weighted by
    # `action_loss_weight`) exists in `camera_diffusion.py` alone. Under any
    # other trainer the head is marked trainable, handed to the optimiser, and
    # given `grad is None` every step -- measured here as `action_head: 0 with
    # a gradient / 6 without`.
    # Only when the head actually exists. `train_control_params: true` expands
    # to every marker name, so stage0 nominally asks for action_out -- but it
    # runs `use_action: false` with no `action_output`, so there is no head for
    # the marker to match and nothing is being quietly left untrained.
    head_exists = bool(cfg.model_kwargs.get("action_output", False))
    if "action_out" in names and head_exists and "camera_diffusion" not in trainer:
        raise SystemExit(
            f"action_out has no supervision under {trainer}: the "
            f"action-prediction loss exists only in camera_diffusion. Use "
            f"--recipe stage1 or sft, or drop action_out from "
            f"train_control_params.")

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
