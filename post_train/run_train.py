import argparse
import math
import os

from omegaconf import OmegaConf
import wandb

from post_train.trainer import CameraDiffusionTrainer

TRAINERS = {
    "camera_diffusion": CameraDiffusionTrainer,
}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tf", action="store_true")
    parser.add_argument("--sp_size", type=int, default=1, help="Sequence parallel size (1=off)")
    parser.add_argument(
        "--allow-small-bsz", action="store_true",
        help="Permit an effective batch below 16. Intended for bounded smoke runs "
             "(a few hundred steps to check the pipeline), where the 16x gradient "
             "accumulation the guard demands would turn 500 steps into many hours. "
             "Do not use it for a real run: the guard exists because DMD and "
             "consistency distillation are unstable at small effective batch.")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "default_config.yaml"))
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.tf = args.tf
    config.sp_size = args.sp_size
    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    # Check effective batch size >= 16
    total_gpus = int(os.environ.get("WORLD_SIZE", 1))
    dp = total_gpus // config.sp_size
    ga = getattr(config, "grad_accum_steps", 1)
    bsz = dp * ga
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[BSZ Check] gpus={total_gpus}, dp={dp}, effective_bsz={bsz}")
        if bsz < 16 and args.allow_small_bsz:
            print(f"[BSZ Check] effective_bsz={bsz} < 16, allowed by --allow-small-bsz. "
                  f"Treat the result as a pipeline check, not a training result.")
        else:
            assert bsz >= 16, (
                f"effective_bsz={bsz} < 16. Suggest: "
                f"--sp_size 1 with grad_accum_steps >= {math.ceil(16 / total_gpus)}, "
                f"or keep --sp_size {config.sp_size} with grad_accum_steps >= {math.ceil(16 / dp)}, "
                f"or --allow-small-bsz for a bounded smoke run"
            )

    if config.trainer not in TRAINERS:
        raise ValueError(f"Unknown trainer: {config.trainer}; "
                         f"expected one of {sorted(TRAINERS)}")
    trainer = TRAINERS[config.trainer](config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
