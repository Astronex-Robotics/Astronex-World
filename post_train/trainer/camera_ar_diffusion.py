"""Camera-controlled AR diffusion trainer (Stage 1).

Standalone trainer that swaps:
1. Model: CameraCausalDiffusion (use_camera=True)
2. Dataset: CameraLatentLMDBDataset (provides viewmats/Ks)
3. train_one_step: passes viewmats/Ks doubled (teacher forcing) to generator_loss

All other logic (save, train loop, FSDP, EMA) inherited from base Trainer.
"""

import gc
import logging

from post_train.data.dataset import cycle, CameraLatentLMDBDataset
from models.wan_wrapper import WanDiffusionWrapper, filter_state_dict_prope_layers
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import math
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job, get_fsdp_process_group, get_sp_data_sampler, get_sp_seed_offset

from post_train.trainer.ar_diffusion import Trainer as _Base


class Trainer(_Base):
    def __init__(self, config):
        self.config = config
        # Preserve the global step when continuing from a trainable-only
        # overlay; checkpoint names, stopping and evaluation gates must not
        # silently restart at zero.
        self.step = int(getattr(config, "ckpt_step", 0) or 0)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job(sp_size=getattr(config, "sp_size", 1))
        from post_train.objectives import CameraCausalDiffusion
        from inference.pipelines import CausalDiffusionInferencePipeline, CausalInferencePipeline
        self._CausalDiffusionInferencePipeline = CausalDiffusionInferencePipeline
        fsdp_pg = get_fsdp_process_group()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + get_sp_seed_offset())

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        self.model = CameraCausalDiffusion(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            # Parameters and gradients on the host: slower, but lets a single
            # 44 GB card run what otherwise needs a second rank to shard.
            cpu_offload=bool(getattr(config, "generator_cpu_offload", False)),
            process_group=fsdp_pg
        )
        # FSDP sharding of the text encoder is optional and off by default for
        # the 2.2 stack. HuggingFace's UMT5EncoderModel puts `embed_tokens` in
        # its own flat parameter, and F.embedding rejects the 1-D shard with
        # "'weight' must be 2-D". It is frozen, called once per step under
        # no_grad, and its outputs are cached by prompt, so there is nothing to
        # gain from sharding it -- only a failure mode.
        if getattr(config, "text_encoder_fsdp", True):
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                # umT5-xxl is ~5.6 GB in bf16 and is used once per step under
                # no_grad. Leaving it resident costs more VRAM than the LoRA
                # optimiser state. The other trainers already offload it; this one
                # read the config key and ignored it.
                cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
                process_group=fsdp_pg,
            )
        else:
            text_device = (
                torch.device("cpu")
                if getattr(config, "text_encoder_cpu_offload", False)
                else self.device
            )
            self.model.text_encoder = self.model.text_encoder.to(
                device=text_device,
                dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        camera_lr = float(getattr(config, "camera_lr", config.lr))
        # An action rate of its own, so the three can be set independently.
        # Freezing the trunk and PRoPE is not the answer -- they have to move
        # with the action branch or the graft ends up mismatched to the features
        # it modulates, which is the failure that took sharpness 872 -> 220 when
        # an embedder trained on one trunk was applied to another. They just
        # have to move slowly enough not to spend what they already are.
        _alr = getattr(config, "action_lr", None)
        action_lr = float(_alr) if _alr is not None else None
        camera_params, action_params, other_params = [], [], []
        for name, param in self.model.generator.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            if "prope_o" in lname:
                camera_params.append(param)
            elif "action_embedder" in lname:
                action_params.append(param)
            else:
                other_params.append(param)
        if bool(getattr(config, "camera_only", False)):
            for param in other_params:
                param.requires_grad_(False)
            other_params = []
        if bool(getattr(config, "action_only", False)):
            for param in camera_params + other_params:
                param.requires_grad_(False)
            camera_params, other_params = [], []
        elif action_lr is None:
            # No separate rate asked for: keep the old single-group behaviour.
            other_params.extend(action_params)
            action_params = []
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": config.lr})
        if camera_params:
            param_groups.append({"params": camera_params, "lr": camera_lr})
        if action_params:
            param_groups.append({"params": action_params,
                                 "lr": action_lr if action_lr is not None else config.lr})
        if self.is_main_process:
            _a = f"{action_lr:g}" if action_lr is not None else f"{config.lr:g}"
            print(f"OPTIMIZER PARAM GROUPS: other={len(other_params)} lr={config.lr:g}, "
                  f"PRoPE={len(camera_params)} lr={camera_lr:g}, "
                  f"action={len(action_params)} lr={_a}")
        optimizer_fused = bool(getattr(config, "optimizer_fused", False))
        optimizer_kwargs = {"fused": True} if optimizer_fused else {
            # Full-parameter 5B training sits close to the 48 GiB limit.
            # The foreach implementation materializes large temporary tensors
            # during the first optimizer step, so allow the memory-lean path.
            "foreach": bool(getattr(config, "optimizer_foreach", True))
        }
        self.generator_optimizer = torch.optim.AdamW(
            param_groups, betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay, **optimizer_kwargs,
        )
        if self.is_main_process:
            print(f"OPTIMIZER IMPLEMENTATION: "
                  f"{'fused' if optimizer_fused else 'foreach=' + str(optimizer_kwargs['foreach'])}")

        from post_train.trainer.control import build_control_dataset, describe_control
        dataset = build_control_dataset(config)
        self.dataset = dataset
        # Tiny exact-replay mixtures need deterministic striding.  With four
        # concatenated rows [case1, case1, case171, case171], shuffle=False
        # gives each of two DP ranks one row from each case.  The old fixed
        # shuffled DistributedSampler (the cycle never calls set_epoch) could
        # pin a rank to one case forever and defeat replay entirely.
        sampler = get_sp_data_sampler(
            dataset,
            shuffle=bool(getattr(config, "data_shuffle", True)),
            drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
            print(describe_control(dataset))
        self.dataloader = cycle(dataloader)

        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue
            self.name_to_trainable_params[rename_param(n)] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            from utils.checkpoint_io import load_checkpoint
            state_dict = load_checkpoint(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            # A full checkpoint predates LoRA wrapping and stores e.g.
            # ``self_attn.q.weight``.  After wrapping, that same frozen tensor
            # lives at ``self_attn.q.base.weight``.  Map only when the original
            # key is absent and the exact base key exists; adapter tensors stay
            # at their zero-init and are the parameters this run will learn.
            if getattr(config.model_kwargs, "lora", None) is not None:
                remapped = {}
                for key, value in state_dict.items():
                    stem, sep, suffix = key.rpartition(".")
                    is_lora_base = sep and suffix in ("weight", "bias") and any(
                        key.endswith(f"{target}.{suffix}")
                        for target in WanDiffusionWrapper.LORA_TARGETS)
                    remapped[
                        f"{stem}.base.{suffix}" if is_lora_base else key
                    ] = value
                state_dict = remapped
            # Limited-layer AC3D layout: an all-layer checkpoint carries extra
            # ``prope_o`` keys for the blocks this run intentionally leaves
            # without a PRoPE branch. Drop them here so the strict unexpected-
            # key guard below still protects against genuinely mismatched
            # checkpoints.
            prope_num_layers = getattr(
                getattr(config, "model_kwargs", None), "prope_num_layers", None)
            state_dict = filter_state_dict_prope_layers(
                state_dict, prope_num_layers)
            # Disabling the action pathway for this run (use_action=False) is
            # intentional, but a checkpoint from an action-enabled run still
            # carries action_embedder tensors this model does not have. Drop
            # exactly those keys so the strict unexpected-key guard below still
            # protects against genuinely mismatched checkpoints.
            if not getattr(config.model_kwargs, "use_action", True):
                dropped_action = [k for k in state_dict
                                  if "action_embedder" in k]
                for k in dropped_action:
                    state_dict.pop(k)
                if dropped_action:
                    print(f"Dropped {len(dropped_action)} action_embedder "
                          "keys: this run has use_action=False")
            # Same for the action-prediction head. The release ships it
            # zero-initialised, so a run that does not train it leaves it out
            # (`action_output: false`) rather than paying for a head whose
            # frozen zero output has no gradient to give -- and then the
            # checkpoint's six head tensors have nowhere to go.
            if not getattr(config.model_kwargs, "action_output", False):
                dropped_head = [k for k in state_dict if "action_head" in k]
                for k in dropped_head:
                    state_dict.pop(k)
                if dropped_head:
                    print(f"Dropped {len(dropped_head)} action_head keys: "
                          "this run has action_output=False")
            # `save_trainable_only` writes just the adapters and the grafted
            # control parameters, so every checkpoint this pipeline produces is
            # a strict *subset* of the model's keys and `strict=True` rejects it
            # outright. Non-strict loading is only safe with the checks below: a
            # bare `strict=False` accepts a checkpoint whose keys match nothing
            # at all and silently trains from the pretrained backbone -- with no
            # camera control, no action pathway, and a log line that looks like
            # a successful resume.
            missing, unexpected = self.model.generator.load_state_dict(
                state_dict, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"checkpoint has {len(unexpected)} keys the model does not, "
                    f"e.g. {list(unexpected)[:5]}; model_kwargs do not describe "
                    "the model this checkpoint came from")
            if not state_dict:
                raise RuntimeError("checkpoint contains no parameters")
            print(f"Loaded {len(state_dict)} tensors; {len(missing)} left at "
                  f"their pretrained values")

        # Optional adapter/control overlay on top of a full base checkpoint.
        # This is used to continue LoRA rollout probes without discarding the
        # full-parameter Stage-1 weights that the adapter was trained against.
        overlay_path = getattr(config, "generator_overlay_ckpt", None)
        if overlay_path:
            print(f"Loading generator overlay from {overlay_path}")
            overlay = torch.load(overlay_path, map_location="cpu")
            overlay = overlay.get(
                "generator", overlay.get("generator_ema", overlay))
            overlay = {
                key.replace("model._fsdp_wrapped_module.", "model.", 1): value
                for key, value in overlay.items()
            }
            prope_num_layers = getattr(
                getattr(config, "model_kwargs", None), "prope_num_layers", None)
            overlay = filter_state_dict_prope_layers(
                overlay, prope_num_layers)
            if not getattr(config.model_kwargs, "use_action", True):
                dropped_action = [k for k in overlay
                                  if "action_embedder" in k]
                for k in dropped_action:
                    overlay.pop(k)
                if dropped_action:
                    print(f"Dropped {len(dropped_action)} action_embedder "
                          "overlay keys: this run has use_action=False")
            missing, unexpected = self.model.generator.load_state_dict(
                overlay, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"overlay has {len(unexpected)} unexpected keys, e.g. "
                    f"{list(unexpected)[:5]}")
            if not overlay:
                raise RuntimeError("generator overlay contains no parameters")

        # Discard whatever the action branch learned before and start it over.
        # The existing weights were fitted on CrossFPS, whose clips pair real
        # stick values with all-identity poses -- commands with no motion under
        # them -- so the branch learned appearance, not displacement. Re-running
        # ActionEmbedder's own init restores its identity-at-init property
        # (to_modulation and embodiment zeroed), so the run starts from base Wan
        # plus a silent action branch rather than from a branch trained to mean
        # the wrong thing.
        if bool(getattr(config, "reinit_action", False)):
            embedder = getattr(self.model.generator.model, "action_embedder", None)
            if embedder is None:
                raise RuntimeError("reinit_action set but the model has no action_embedder")
            for module in embedder.proj:
                if hasattr(module, "weight") and module.weight.dim() > 1:
                    torch.nn.init.xavier_uniform_(module.weight)
                if getattr(module, "bias", None) is not None:
                    torch.nn.init.zeros_(module.bias)
            torch.nn.init.zeros_(embedder.to_modulation.weight)
            torch.nn.init.zeros_(embedder.to_modulation.bias)
            if embedder.embodiment is not None:
                torch.nn.init.zeros_(embedder.embodiment.weight)
            print("[reinit_action] action branch reset to identity-at-init")
            print(f"Overlaid {len(overlay)} tensors; {len(missing)} base tensors unchanged")

        if bool(getattr(config, "reset_action_output", False)):
            reset = 0
            with torch.no_grad():
                for name, param in self.model.generator.named_parameters():
                    if "action_embedder.to_modulation" in name:
                        param.zero_()
                        reset += 1
            print(f"Reset {reset} camera-plan output tensors to identity")

        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None
        self.delta_mean = None
        self.rtf_ema_ratio = getattr(self.config, "rtf_ema_ratio", 0.9)
        self.eval_interval = getattr(self.config, "eval_interval", 0)
        self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
        self.eval_init = getattr(self.config, "eval_num_init_frames", 3)
        self.rtf_single_gpu_batch = getattr(self.config, "rtf_single_gpu_batch", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
        if self.eval_interval:
            self.pipeline = self._CausalDiffusionInferencePipeline(config, device=self.device)
            self.pipeline.generator = self.model.generator
            self.pipeline.text_encoder = self.model.text_encoder

    def train_one_step(self, batch):
        # Save cadence comes from the config. Hardcoding 1 writes a full
        # checkpoint every step, which for a 5B model is ~10 GB per step.
        self.log_iters = int(getattr(self.config, "save_iters", self.config.log_iters))

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:
            clean_latent = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
        else:
            frames = batch["frames"].to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(frames).to(device=self.device, dtype=self.dtype)
        requested_frames = int(self.config.image_or_video_shape[1])
        clean_latent = clean_latent[:, :requested_frames]
        image_latent = clean_latent[:, 0:1, ]

        # `feed_camera: false` withholds the camera stream rather than freezing
        # PRoPE. It is for datasets whose poses carry nothing: CrossFPS ships
        # [0,0,0, 0,0,0,1] for every frame of every clip, so feeding it trains
        # PRoPE to expect a constant, and 295 steps of that is why the action
        # checkpoint answers a real trajectory worse than the trunk it came
        # from. Withholding also drops the two PRoPE KV caches, ~5.7 GB.
        if bool(getattr(self.config, "feed_camera", True)):
            viewmats = batch["viewmats"][:, :requested_frames].to(
                device=self.device, dtype=self.dtype)
            Ks = batch["Ks"][:, :requested_frames].to(
                device=self.device, dtype=self.dtype)
        else:
            viewmats = None
            Ks = None

        # Actions are per latent frame like the poses, so they are cut to the
        # same length. Left full, a clip longer than the config reaches the
        # model with more actions than frames and fails in the wrapper.
        if "actions" in batch:
            batch = dict(batch)
            batch["actions"] = batch["actions"][:, :requested_frames]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # With the frozen text encoder resident on CPU, only its compact output
        # crosses to the GPU; keeping UMT5 itself there frees enough memory for
        # the full 5B AdamW states.
        conditional_dict = {
            k: (v.to(device=self.device, dtype=self.dtype) if torch.is_tensor(v) else v)
            for k, v in conditional_dict.items()
        }
        unconditional_dict = {
            k: (v.to(device=self.device, dtype=self.dtype) if torch.is_tensor(v) else v)
            for k, v in unconditional_dict.items()
        }

        # Camera / action / teacher-prompt channels. Both dicts get them: CFG
        # must vary the text only (see post_train/trainer/control.py).
        from post_train.trainer.control import inject_control
        inject_control(conditional_dict, unconditional_dict, batch,
                       self.device, self.dtype, self.config, text_prompts,
                       event_encoder=getattr(self.model, "event_encoder", None))

        # Chained forward training: supervise every window of the chain, not
        # one sampled window per step.
        #
        # Memory Forcing's chain loss averages the denoising loss over all
        # windows, where each window's context holds ground truth for frames
        # not yet predicted and the model's own predictions for frames it
        # already produced. The existing loss already builds that context --
        # it rolls blocks 0..target-1 from the model's own samples -- but it
        # charges only the one sampled target, so most of the chain is
        # generated and then never scored.
        #
        # Each target is a separate call with its own backward rather than one
        # summed graph: the blocks would otherwise all carry activations at
        # once, and this run already sits near the card's ceiling. Gradients
        # accumulate into the same buffers and are clipped and stepped once,
        # which is the same update a summed loss would have produced.
        chained = bool(getattr(self.config, "chained_forward", False))
        if chained:
            n_blocks = (int(self.config.num_training_frames)
                        // int(self.config.num_frame_per_block))
            targets = list(range(1, max(2, n_blocks)))
        else:
            targets = [None]

        self.generator_optimizer.zero_grad()
        generator_loss = None
        chain_log = {}
        for tgt in targets:
            if tgt is not None:
                self.model.args.scheduled_target_block = tgt
            loss_b, log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent,
                viewmats=viewmats,
                Ks=Ks
            )
            (loss_b / len(targets)).backward()
            generator_loss = (loss_b.detach() if generator_loss is None
                              else generator_loss + loss_b.detach())
            # Average every scalar over the chain, not just keep the last
            # window's. Overwriting log_dict per target made the component
            # losses describe block 4 alone while `loss` described the chain,
            # so the two could not be read against each other.
            if len(targets) > 1:
                for k, v in log_dict.items():
                    if not torch.is_tensor(v) or v.numel() != 1:
                        continue
                    chain_log[k] = chain_log.get(k, 0.0) + v.detach()
        if len(targets) > 1:
            generator_loss = generator_loss / len(targets)
            self.model.args.scheduled_target_block = -1
            for k, v in chain_log.items():
                log_dict[k] = v / len(targets)

        # Second forward/backward, outside the training clip.
        #
        # It cannot ride along with the pass above. The backbone
        # gradient-checkpoints, so its blocks are re-run during backward; a
        # rollout writing into the same kv/PRoPE caches while that graph is
        # pending leaves the recomputation reading a cache that has moved on.
        # Running here also keeps the two memory peaks from adding -- the pass
        # above already fills 43.1 GiB of a 44.39 GiB card -- because its graph
        # is freed by the backward that just returned. Gradients accumulate
        # into the same .grad buffers and are clipped and stepped together.
        depth_loss = self.model.depth_spectrum_loss(
            conditional_dict=conditional_dict,
            clean_latent=clean_latent,
            unconditional_dict=unconditional_dict,
            viewmats=viewmats,
            Ks=Ks,
        )
        if depth_loss is not None:
            depth_loss.backward()
            log_dict["depth_spectrum_loss"] = depth_loss.detach()
        if self.is_main_process and not getattr(self, "_first_bp_logged", False):
            print("[Trainer Entry] post_train/trainer/camera_ar_diffusion.py :: Trainer.train_one_step (first BP done)", flush=True)
            self._first_bp_logged = True
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()

        self.step += 1

        # Any scalar the loss returns gets logged. `generator_loss` is a sum --
        # flow matching plus, when the action head is on, its MSE -- and a sum
        # is the one number that cannot tell you which term moved. A run whose
        # entire purpose is the action head reporting only the total is a run
        # you cannot read.
        extra_scalars = {
            k: v.item() for k, v in log_dict.items()
            if torch.is_tensor(v) and v.dim() == 0
        }
        if self.is_main_process and not self.disable_wandb:
            wandb.log({
                "generator_loss": generator_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                **extra_scalars,
            }, step=self.step)
        if self.is_main_process:
            console_every = int(getattr(self.config, "console_log_every", 0))
            if console_every > 0 and self.step % console_every == 0:
                tail = "".join(f"  {k} {v:.6f}" for k, v in extra_scalars.items())
                print(
                    f"[camera-ar] step {self.step:6d}  "
                    f"loss {generator_loss.item():.6f}  "
                    f"grad_norm {generator_grad_norm.item():.6f}{tail}",
                    flush=True,
                )

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()
