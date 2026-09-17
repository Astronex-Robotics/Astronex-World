import gc
import logging

from post_train.data.dataset import cycle, LatentLMDBDataset
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import math
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job, get_fsdp_process_group, get_sp_data_sampler, get_sp_seed_offset

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job(sp_size=getattr(config, "sp_size", 1))
        # Import objectives/pipelines AFTER distributed init so causal_model.py sees CleanCode SP infra
        from post_train.objectives import CausalDiffusion
        from inference.pipelines import CausalDiffusionInferencePipeline, CausalInferencePipeline
        self._CausalDiffusionInferencePipeline = CausalDiffusionInferencePipeline
        fsdp_pg = get_fsdp_process_group()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
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

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
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
            self.model.text_encoder = self.model.text_encoder.to(
                device=self.device,
                dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))
       
        self.dataset = dataset
        sampler = get_sp_data_sampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
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
            self.model.generator.load_state_dict(state_dict, strict=True)

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None
        self.delta_mean = None
        self.rtf_ema_ratio = getattr(self.config, "rtf_ema_ratio", 0.9) 
        self.eval_interval = getattr(self.config, "eval_interval", 0)      # 0 => disable
        self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
        self.eval_init = getattr(self.config, "eval_num_init_frames", 3)
        self.rtf_single_gpu_batch = getattr(self.config, "rtf_single_gpu_batch", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
        if self.eval_interval:
            self.pipeline = self._CausalDiffusionInferencePipeline(config, device=self.device)
            self.pipeline.generator = self.model.generator
            self.pipeline.text_encoder = self.model.text_encoder
            
    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

        trainable_only = bool(getattr(self.config, "save_trainable_only", False))
        from post_train.trainer.control import TRAINABLE_MARKERS as markers
        if trainable_only and self.is_main_process:
            generator_state_dict = {
                k: v for k, v in generator_state_dict.items()
                if any(marker in k for marker in markers)
            }
            if not generator_state_dict:
                raise RuntimeError(
                    "save_trainable_only matched no LoRA/control tensors")

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        # Every rank participates in fsdp_state_dict and must agree that this
        # step has been saved. Setting this only on rank 0 makes other ranks
        # enter a second gather at max_step after rank 0 has already returned.
        self._last_saved_step = self.step
        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            final_path = os.path.join(checkpoint_dir, "model.pt")
            tmp_path = final_path + ".tmp"
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, final_path)
            print("Model saved to", final_path)

    def train_one_step(self, batch):
        # Save cadence comes from the config. Hardcoding 1 writes a full
        # checkpoint every step, which for a 5B model is ~10 GB per step.
        self.log_iters = int(getattr(self.config, "save_iters", self.config.log_iters))

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["clean_latent"].to(
                device=self.device, dtype=self.dtype)
        else:  # encode raw video to latent
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)
           
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
        image_latent = clean_latent[:, 0:1, ]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts) 
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent
        )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Step 4: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            console_every = int(getattr(self.config, "console_log_every", 0))
            if console_every > 0 and self.step % console_every == 0:
                print(
                    f"[ar] step {self.step:6d}  "
                    f"loss {generator_loss.item():.6f}  "
                    f"grad_norm {generator_grad_norm.item():.6f}",
                    flush=True,
                )

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()


    def train(self):
        # `max_step` lets a run be bounded ("500 steps, then evaluate") instead
        # of only ever being stopped by killing it.
        max_step = int(getattr(self.config, "max_step", 0))

        while True:
            if max_step and self.step >= max_step:
                if self.is_main_process:
                    print(f"[Trainer] reached max_step={max_step}, saving and stopping",
                          flush=True)
                if (not self.config.no_save and
                        getattr(self, "_last_saved_step", None) != self.step):
                    self.save()
                return
            batch = next(self.dataloader)
            self.train_one_step(batch)
                
            save_iters = int(getattr(self.config, "save_iters", self.config.log_iters))
            if (not self.config.no_save) and self.step % save_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
