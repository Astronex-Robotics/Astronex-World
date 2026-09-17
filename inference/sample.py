import ast
import argparse
import torch
import torch.nn.functional as F
import sys
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
import imageio
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from PIL import Image
import json

from post_train.data.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from utils.camera_trajectory import parse_trajectory

from inference.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

# Released Astronex-World weights ship as .safetensors, which is flat and so
# carries its section in the key prefix; this restores the nested form the
# loading code below expects. .pt is unaffected.
from utils.checkpoint_io import load_checkpoint  # noqa: E402


class SingleImageTextDataset(Dataset):
    """One reference image plus one prompt, for quick I2V smoke tests."""

    def __init__(self, image_path, prompt, transform=None):
        self.image_path = image_path
        self.prompt = prompt
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "prompts": self.prompt,
            "idx": idx,
        }


class WBenchManifestDataset(Dataset):
    """Batch I2V cases while keeping one model resident on each GPU."""

    def __init__(self, manifest_path, transform=None):
        with open(manifest_path, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        item = {
            "prompts": row["prompt"],
            "idx": idx,
            "output_name": row["output_name"],
            "num_output_frames": int(row["num_output_frames"]),
            "camera_path": row.get("camera_path", ""),
            # Per-row, because WBench turns differ: an event_edit turn carries
            # text for the event channel and a navigation turn carries none, and
            # the two are interleaved inside a single case. The CLI's
            # --event_prompt is one value for the whole invocation, which cannot
            # express that. "" means no event, matching the no-event path
            # exactly rather than injecting an empty string.
            "event_prompt": row.get("event_prompt", ""),
            # Per-row too: a timed turn introduces its text partway through,
            # while a navigation turn has no event at all and must take the
            # frame-0 path.
            "event_start_frame": int(row.get("event_start_frame", 0) or 0),
        }
        # WBench/VBench-I2V rows have a reference image.  VBench 1.0 is T2V,
        # but uses the same manifest machinery to preserve exact output names,
        # per-sample seeds and lengths without loading thousands of models.
        if row.get("image_path"):
            image = Image.open(row["image_path"]).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            item["image"] = image
        return item


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--overlay_checkpoint_path", type=str, default=None,
                    help="Optional trainable-only checkpoint applied after --checkpoint_path")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--image_path", type=str, default=None,
                    help="Reference image for I2V; overrides --data_path dataset layout")
parser.add_argument("--wbench_manifest", type=str, default=None,
                    help="JSONL batch manifest; keeps the model resident across WBench cases")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=20, help="Number of overlap frames between sliding windows")
parser.add_argument("--sampling_steps", type=int, default=None,
                    help="Override diffusion/UniPC sampling steps from the config")
parser.add_argument("--camera_guidance_scale", type=float, default=None,
                    help="Classifier-free guidance on the CAMERA condition, "
                         "the third term of Camera Motion Guidance "
                         "(arXiv 2410.10802). Text CFG leaves the camera in "
                         "both branches, so it cancels out of the difference "
                         "and only the text is ever amplified; this scales a "
                         "camera-conditioned minus camera-neutral prediction "
                         "instead. 0 disables it, and each step above 0 costs "
                         "one extra forward pass.")
parser.add_argument("--guidance_scale", type=float, default=None,
                    help="Override classifier-free guidance scale")
parser.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                    help="Override a TOP-LEVEL config entry, e.g. "
                         "--set prope_pin_sink=true. Distinct from --mk, which "
                         "targets model_kwargs: the pipeline reads some flags "
                         "off the config root and the wrapper rejects unknown "
                         "model_kwargs outright, so the two are not "
                         "interchangeable.")
parser.add_argument("--timestep_shift", type=float, default=None,
                    help="Override the sampler's timestep_shift (the top-level "
                         "one, not model_kwargs.timestep_shift, which is the "
                         "value the checkpoint was trained at). Larger puts "
                         "more of the step budget in the high-noise region.")
parser.add_argument("--latent_height", type=int, default=None,
                    help="Override latent output height (44 corresponds to 704 pixels)")
parser.add_argument("--latent_width", type=int, default=None,
                    help="Override latent output width (80 corresponds to 1280 pixels)")
parser.add_argument("--num_frame_per_block", type=int, default=None,
                    help="Inference override for the causal block size")
parser.add_argument("--local_attn_size", type=int, default=None,
                    help="Inference override for the sink+recent KV window, in "
                         "latent frames (e.g. 9 = 1 sink + full previous block "
                         "+ current block at block size 4)")
parser.add_argument("--latent_stitch_strength", type=float, default=None,
                    help="Blend causal latent chunks toward the previous chunk's "
                         "terminal motion trend (0 disables, recommended probe 0.7)")
parser.add_argument("--latent_stitch_preserve_mean", action="store_true",
                    help="Remove each channel's spatial mean from latent stitch "
                         "corrections so chunk continuity does not shift exposure")
parser.add_argument("--latent_stitch_highpass_kernel", type=int, default=None,
                    help="Remove spatial frequencies below this odd pooling kernel "
                         "from stitch corrections (0/1 disables)")
parser.add_argument("--latent_temporal_smooth", type=float, default=None,
                    help="Suppress second-order latent temporal jitter while "
                         "preserving constant-velocity motion (probe: 0.12)")
parser.add_argument("--rgb_chroma_stabilize_strength", type=float, default=None,
                    help="Correct isolated decoded RGB colour jumps without "
                         "spatial filtering (0 disables; validated value 0.8)")
parser.add_argument("--rgb_chroma_stabilize_fade", type=int, default=None,
                    help="Temporal radius of the triangular RGB-mean smoother")
parser.add_argument("--window_latent_frames", type=int, default=0,
                    help="Reset causal KV after this many total latent frames, "
                         "conditioning the next window on the previous final latent")
parser.add_argument("--window_crossfade_rgb", type=int, default=4,
                    help="RGB frames cross-faded at long-window reset boundaries")
parser.add_argument("--window_overlap_latents", type=int, default=4,
                    help="Clean latent overlap used to condition each reset window")
parser.add_argument("--window_color_match", action="store_true",
                    help="Match each reset window's RGB mean/std to its true "
                         "overlap with the previous window")
parser.add_argument("--window_color_match_strength", type=float, default=0.25,
                    help="Fraction of overlap RGB color correction to apply")
# The causal pipeline prefills the conditioning image into the KV cache. With
# one frame that prefill is its own block, which shifts every later block one
# frame off the boundaries the model was trained on. Repeating the image to
# fill a whole block restores the training alignment at the cost of starting
# the clip with N identical frames -- an ablation, not a default.
parser.add_argument("--cond_repeat", type=int, default=1,
                    help="Repeat the conditioning latent this many times")
parser.add_argument("--ref_cond_scale", type=float, default=None,
                    help="Scale the persistent reference-frame attention values "
                         "after clean cache prefill (1 keeps trained behaviour)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--sp_size", type=int, default=1, help="Sequence parallel size (1=disabled)")
parser.add_argument("--event_prompt", type=str, default=None,
                    help="Independent event-text channel (e.g. 'the arm picks up the scissors'). "
                         "Routed through its own cross-attention, not merged into the caption, "
                         "so it can be set or dropped without touching the prompt.")
parser.add_argument("--profile", action="store_true",
                    help="Print the initialisation / diffusion / VAE time split. The "
                         "pipeline has always measured this; there was simply no flag "
                         "to turn it on, so the breakdown was unreachable from the CLI.")
parser.add_argument("--save_latents", action="store_true",
                    help="Also write the generated latents next to the mp4. The decoded "
                         "video is not a clean signal for ablations: the VAE decoder is "
                         "run over the whole clip, so its temporal receptive field spreads "
                         "a change in one latent frame across earlier RGB frames. Compare "
                         "latents when you need to know what the model actually did.")
parser.add_argument("--fps", type=int, default=16,
                    help="Frame rate written into the mp4. 16 is this repo's "
                         "own convention and stays the default; WBench scores "
                         "against FPS=24 (src/models/camera/poses.py), and a "
                         "clip written at 16 plays 1.5x slow, which is not a "
                         "cosmetic difference -- MegaSAM reads the camera "
                         "trajectory back out of the timing.")
parser.add_argument("--event_start_frame", type=int, default=0,
                    help="Latent frame the event prompt is injected from (default 0 = "
                         "the whole rollout). Chunks before it are bit-identical to a "
                         "run with no event, so the scene is established first and the "
                         "event lands mid-video.")
parser.add_argument("--action_tokens", type=str, default=None,
                    help="Per-latent-frame action names driving the CrossFPS "
                         "action embedder, comma-separated with '*N' repeats "
                         "like --trajectory (e.g. 'W*8,A*8'). Names come from "
                         "utils.action_encoding.WBENCH_ACTION. This is a "
                         "second, independent control path: PRoPE says where "
                         "the camera is, the embedder says what was commanded.")
parser.add_argument("--action_scale", type=float, default=1.0,
                    help="Stick deflection for --action_tokens. 1.0 is a fully "
                         "pushed stick; the training data's per-clip means run "
                         "to about 0.9, so values above 1 are outside it.")
parser.add_argument("--embodiment_id", type=int, default=None,
                    help="Row of action_embedder.embodiment (CrossFPS titles).")
parser.add_argument("--trajectory", type=str, default=None, help="Camera trajectory string (e.g., 'w*19' for camera control)")
parser.add_argument("--trajectory_speed", type=float, default=None,
                    help="Scale camera translation relative to the first pose; rotations are unchanged")
parser.add_argument("--trajectory_path", type=str, default=None, help="Path to trajectory file (one trajectory string per line, aligned with data_path)")
parser.add_argument("--camera_diag", action="store_true",
                    help="Print per-block causal PRoPE contribution norms")
parser.add_argument("--camera_diag_path", type=str, default=None,
                    help="Optional torch file for direction-sensitive PRoPE diagnostics")
parser.add_argument("--reverse_time_attention", action="store_true",
                    help="Reverse inputs and outputs so causal attention becomes future-only in output time")
# LongLive identity / foreground routing overrides.  The defaults (None)
# leave whatever is set in config.model_kwargs untouched; the config's
# defaults in turn match pre-fix behaviour (pooled routing / no boost).
parser.add_argument("--history_per_token_routing",
                    choices=["auto", "on", "off"], default=None,
                    help="Override history block routing to the per-token "
                         "median-variant (on) / keep pooled LongLive baseline "
                         "(off).  auto = use config value.")
parser.add_argument("--history_sink_boost", type=float, default=None,
                    help="Multiplicative boost applied to the persistent "
                         "reference/sink frame contribution inside history "
                         "cross-attention.  1 = no boost; 1.35 ~ restores "
                         "roughly the extra attention share bidirectional "
                         "gives the opening frame on a library-hall shot.")
parser.add_argument("--mk", action="append", default=None, metavar="KEY=VALUE",
                    help="Override an arbitrary model_kwargs entry, e.g. "
                         "--mk prope_exclude_sink=true.  Values are parsed as "
                         "Python literals, falling back to the raw string.  "
                         "Repeatable.  Ablation-only: anything worth keeping "
                         "belongs in the eval config.")
parser.add_argument("--history_topk", type=int, default=None,
                    help="Override history_cross_topk in model_kwargs.")
parser.add_argument("--history_scale", type=float, default=None,
                    help="Override history_cross_scale in model_kwargs.")
parser.add_argument("--action_keys", action="store_true",
                    help="Read --action_tokens through the KEYBOARD table in "
                         "utils.action_encoding instead of WBENCH_ACTION, "
                         "so a segment is the key a player presses "
                         "('w,a,s,d,arrowleft,...'). Both tables address the "
                         "same stick lanes; this is the keyboard-to-stick "
                         "binding a live session would use.")
parser.add_argument("--history_repeat", type=int, default=None,
                    help="Override model_kwargs.history_attention_repeat. Like "
                         "--rollout_repeat it lowers the sink's share of the "
                         "softmax, but repeats the already-denoised history "
                         "instead of the block being denoised, so it does not "
                         "amplify noise. 1 is off.")
parser.add_argument("--rollout_repeat", type=int, default=None,
                    help="Override model_kwargs.rollout_attention_repeat. "
                         "Repeats the current block in attention, which divides "
                         "the softmax weight of sink and history by this factor "
                         "without removing them: the reference keeps guiding, "
                         "the rollout stops being overruled by it. 1 is off.")
parser.add_argument("--sink_size", type=int, default=None,
                    help="Override model_kwargs.sink_size (persistent KV "
                         "sink frames pinned to the start of attention).")
parser.add_argument("--text_cross_decay_start", type=int, default=None,
                    help="Start frame of text cross-attention decay.  "
                         "Pass a large value (e.g. 10_000_000) to disable it "
                         "and keep prompt-level identity anchors live for "
                         "small foreground objects.")
parser.add_argument("--text_cross_late_scale", type=float, default=None,
                    help="Residual scale of text cross-attention after "
                         "decay.  Values > 1.0 help keep 'same lantern / "
                         "same hand' constraints after block ~12.")
args = parser.parse_args()
if args.camera_diag:
    os.environ["WAN_CAMERA_DIAG"] = "1"

# Initialize distributed inference
# IMPORTANT: distributed init MUST happen before importing pipeline modules,
# because causal_model.py checks for CleanCode SP infra at import time.
if args.sp_size > 1:
    # SP mode requires torchrun with nproc_per_node >= sp_size
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size_env >= args.sp_size, (
        f"SP requires at least {args.sp_size} processes, but WORLD_SIZE={world_size_env}. "
        f"Launch with: torchrun --nproc_per_node={args.sp_size} -m inference.sample ... --sp_size {args.sp_size}"
    )
    from utils.distributed import launch_distributed_job, get_sp_seed_offset
    launch_distributed_job(backend="nccl", sp_size=args.sp_size)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
elif "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

# Seed: under SP, ranks in the same SP group must share the same seed
if args.sp_size > 1:
    set_seed(args.seed + get_sp_seed_offset())
else:
    set_seed(args.seed)

# Refresh gpu device handle (inference.memory.gpu is captured at import time
# before distributed init sets the correct CUDA device)
from inference import memory as _mem
_mem.gpu = torch.device(f'cuda:{local_rank}')
gpu = _mem.gpu

# ASTRONEX_VRAM_LIMIT_GB caps what this process may allocate, so a 44 GB card
# behaves like a smaller one: an allocation past the cap raises OOM exactly as
# it would on the real card.
_vram_limit = os.environ.get("ASTRONEX_VRAM_LIMIT_GB")
if _vram_limit:
    _total = torch.cuda.get_device_properties(gpu).total_memory / (1024 ** 3)
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, float(_vram_limit) / _total), gpu)
    print(f"[vram] capped at {float(_vram_limit):.1f} GB of {_total:.1f} GB")
_free_vram = get_cuda_free_memory_gb(gpu)
if _vram_limit:
    _free_vram = min(_free_vram, float(_vram_limit))
print(f'Free VRAM {_free_vram} GB')
low_memory = (
    os.environ.get("ASTRONEX_DISABLE_LOW_MEMORY") != "1"
    and _free_vram < 40
)

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "default_config.yaml"))
config = OmegaConf.merge(default_config, config)
if args.sampling_steps is not None:
    config.sampling_steps = args.sampling_steps
if args.guidance_scale is not None:
    config.guidance_scale = args.guidance_scale
if args.camera_guidance_scale is not None:
    config.camera_guidance_scale = args.camera_guidance_scale
    print(f"[override] camera_guidance_scale = {config.camera_guidance_scale}")
if args.timestep_shift is not None:
    config.timestep_shift = args.timestep_shift
    print(f"[override] timestep_shift = {config.timestep_shift}")
if args.latent_height is not None:
    config.image_or_video_shape[3] = args.latent_height
if args.latent_width is not None:
    config.image_or_video_shape[4] = args.latent_width
if args.num_frame_per_block is not None:
    config.num_frame_per_block = args.num_frame_per_block
if args.local_attn_size is not None:
    config.model_kwargs.local_attn_size = args.local_attn_size
if args.latent_stitch_strength is not None:
    config.latent_stitch_strength = args.latent_stitch_strength
if args.latent_stitch_preserve_mean:
    config.latent_stitch_preserve_mean = True
if args.latent_stitch_highpass_kernel is not None:
    config.latent_stitch_highpass_kernel = args.latent_stitch_highpass_kernel
if args.latent_temporal_smooth is not None:
    config.latent_temporal_smooth = args.latent_temporal_smooth
if args.rgb_chroma_stabilize_strength is not None:
    config.rgb_chroma_stabilize_strength = args.rgb_chroma_stabilize_strength
if args.rgb_chroma_stabilize_fade is not None:
    config.rgb_chroma_stabilize_fade = args.rgb_chroma_stabilize_fade
if args.ref_cond_scale is not None:
    config.ref_cond_scale = args.ref_cond_scale
if args.trajectory_speed is None:
    args.trajectory_speed = float(config.get("trajectory_speed", 1.0))
if args.window_latent_frames == 0:
    args.window_latent_frames = int(config.get("window_latent_frames", 0))
if args.window_overlap_latents == 4:
    args.window_overlap_latents = int(config.get(
        "window_overlap_latents", args.window_overlap_latents))
if args.window_crossfade_rgb == 4:
    args.window_crossfade_rgb = int(config.get(
        "window_crossfade_rgb", args.window_crossfade_rgb))
if not args.window_color_match:
    args.window_color_match = bool(config.get("window_color_match", False))
if args.window_color_match_strength == 0.25:
    args.window_color_match_strength = float(config.get(
        "window_color_match_strength", args.window_color_match_strength))

# ----- LongLive identity / foreground overrides (apply BEFORE wrapper build)
for _entry in (args.set or []):
    _key, _, _raw = _entry.partition("=")
    try:
        _value = ast.literal_eval(_raw.capitalize() if _raw.lower() in
                                  ("true", "false", "none") else _raw)
    except (ValueError, SyntaxError):
        _value = _raw
    config[_key] = _value
    print(f"[set-override] {_key} = {_value!r}")
for _entry in (args.mk or []):
    _key, _, _raw = _entry.partition("=")
    try:
        _value = ast.literal_eval(_raw.capitalize() if _raw.lower() in
                                  ("true", "false", "none") else _raw)
    except (ValueError, SyntaxError):
        _value = _raw
    config.model_kwargs[_key] = _value
    print(f"[mk-override] model_kwargs.{_key} = {_value!r}")
if args.sink_size is not None:
    config.model_kwargs.sink_size = int(args.sink_size)
if args.rollout_repeat is not None:
    config.model_kwargs.rollout_attention_repeat = int(args.rollout_repeat)
if args.history_repeat is not None:
    config.model_kwargs.history_attention_repeat = int(args.history_repeat)
if args.history_topk is not None:
    config.model_kwargs.history_cross_topk = int(args.history_topk)
if args.history_scale is not None:
    config.model_kwargs.history_cross_scale = float(args.history_scale)
if args.history_per_token_routing in ("on", "off"):
    config.model_kwargs.history_cross_per_token_routing = (
        args.history_per_token_routing == "on")
if args.history_sink_boost is not None:
    config.model_kwargs.history_cross_sink_boost = float(args.history_sink_boost)
if args.text_cross_decay_start is not None:
    config.model_kwargs.text_cross_attn_decay_start = int(
        args.text_cross_decay_start)
if args.text_cross_late_scale is not None:
    config.model_kwargs.text_cross_attn_late_scale = float(
        args.text_cross_late_scale)
# Import pipeline AFTER distributed init so causal_model.py sees CleanCode SP infra
from inference.pipelines import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
    BidirectionalDiffusionInferencePipeline,
    BidirectionalInferencePipeline,
)

# Initialize pipeline
is_causal = config.get('causal', True)

if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    if is_causal:
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = BidirectionalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    if is_causal:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)
    else:
        pipeline = BidirectionalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = load_checkpoint(args.checkpoint_path, map_location="cpu")
    # Honour --use_ema instead of silently preferring the EMA weights.
    #
    # This used to try 'generator_ema' first and fall back, which made
    # --use_ema decorative and meant every evaluation of a checkpoint that
    # happened to carry an EMA entry sampled the EMA -- including EMA entries
    # that were never updated. A run whose trainer never called
    # EMA_FSDP.update() writes a `generator_ema` holding the model as
    # constructed: LoRA B still zero, so the adapters are an exact no-op, and
    # prope_o still zero, so the camera pathway contributes nothing. Sampling
    # that is sampling pristine pretrained Wan2.2, and it looks like a plausible
    # video, so nothing about the output says the trained weights were skipped.
    key = 'generator_ema' if args.use_ema else 'generator'
    if key not in state_dict and 'generator' not in state_dict and 'generator_ema' not in state_dict:
        # Bare merged state dict (e.g. logs/merged/pcd100.pt): the checkpoint
        # already holds the full model under `model.*` keys.
        print(f"[ckpt] bare state dict, loading directly from {args.checkpoint_path}")
        gen_sd = state_dict
        key = "state_dict"
    elif key not in state_dict:
        fallback = 'generator' if key == 'generator_ema' else 'generator_ema'
        if fallback not in state_dict:
            raise KeyError(
                f"checkpoint has neither '{key}' nor '{fallback}': "
                f"{list(state_dict.keys())[:5]}")
        print(f"[ckpt] '{key}' not in checkpoint, using '{fallback}'")
        key = fallback
        gen_sd = state_dict[key]
    else:
        gen_sd = state_dict[key]
    print(f"[ckpt] loading '{key}' ({len(gen_sd)} tensors) from {args.checkpoint_path}")
    
    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

if args.overlay_checkpoint_path:
    overlay = load_checkpoint(args.overlay_checkpoint_path, map_location="cpu")
    key = "generator" if "generator" in overlay else next(iter(overlay))
    overlay_sd = {
        k.replace("model._fsdp_wrapped_module.", "model.", 1): v
        for k, v in overlay[key].items()
    }
    incompatible = pipeline.generator.load_state_dict(overlay_sd, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"overlay has unexpected keys: {incompatible.unexpected_keys[:5]}")
    print(f"[ckpt] overlaid '{key}' ({len(overlay_sd)} tensors) from "
          f"{args.overlay_checkpoint_path}")

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
vae_device = torch.device(os.environ.get("WAN_VAE_DEVICE", str(gpu)))
pipeline.vae.to(device=vae_device)
pipeline.vae_device = vae_device if vae_device != gpu else None
if pipeline.vae_device is not None:
    print(f"[device] generator={gpu}, vae={pipeline.vae_device}")


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    # Keep the reference image on the same spatial grid as the requested
    # latent output.  The previous fixed 480x832 resize made the documented
    # 720p (44x80 latent) preset fail when concatenating reference_latent with
    # the 720p noisy input.
    target_height = int(config.image_or_video_shape[3]) * 16
    target_width = int(config.image_or_video_shape[4]) * 16
    transform = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    if args.wbench_manifest:
        dataset = WBenchManifestDataset(args.wbench_manifest, transform=transform)
    elif args.image_path:
        with open(args.data_path, encoding="utf-8") as _f:
            prompt = _f.read().strip()
        dataset = SingleImageTextDataset(
            args.image_path, prompt, transform=transform)
    else:
        dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = (WBenchManifestDataset(args.wbench_manifest)
               if args.wbench_manifest
               else TextDataset(prompt_path=args.data_path))
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized() and args.sp_size <= 1:
    # Standard DP: split prompts across ranks
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
elif dist.is_initialized() and args.sp_size > 1:
    # SP mode: use SP-aware sampler so ranks in the same SP group get the same data
    from utils.distributed import get_sp_data_sampler
    sampler = get_sp_data_sampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# Load per-prompt trajectory list if provided
trajectory_list = None
if args.trajectory_path:
    with open(args.trajectory_path, encoding="utf-8") as _f:
        trajectory_list = [line.strip() for line in _f if line.strip()]
    assert len(trajectory_list) >= num_prompts, (
        f"trajectory_path has {len(trajectory_list)} lines but need >= {num_prompts} prompts"
    )

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


# Latency bookkeeping (rank 0 only; first prompt is recorded as None to skip warmup).
# All pipelines expose `last_chunk0_latency`: time from sampling start to the first
# denoised latent ready, EXCLUDING VAE decode — matches the reported latency definition.
chunk0_latencies = []


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    
    
    if args.i2v:
        if config.num_frame_per_block != 1:
            print("I2V uses simple first-frame conditioning "
                  f"(num_frame_per_block={config.num_frame_per_block})")
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        output_name = (batch.get('output_name', [f'{prompt[:100]}.mp4'])[0]
                       if args.wbench_manifest else f'{prompt[:100]}.mp4')
        output_path = os.path.join(args.output_folder, output_name)
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(
            image.to(device=vae_device)).to(
            device=device, dtype=torch.bfloat16)
        if args.cond_repeat > 1:
            initial_latent = initial_latent.repeat(
                1, args.cond_repeat, *([1] * (initial_latent.dim() - 2)))
        prompts = [prompt] 
        sample_frames = (int(batch['num_output_frames'].item())
                         if args.wbench_manifest else args.num_output_frames)
        # WBench manifests store the number of generated latents *after* the
        # reference latent. Causal I2V concatenates that reference separately,
        # while bidirectional I2V replaces frame 0 inside the sampled tensor.
        # Give the latter one extra slot so video latents and camera poses have
        # identical full-sequence lengths.
        if args.wbench_manifest and not is_causal:
            sample_frames += 1
        sampled_noise = torch.randn(
            [1, sample_frames,
             config.image_or_video_shape[2],
             config.image_or_video_shape[3],
             config.image_or_video_shape[4]],
            device=device, dtype=torch.bfloat16,
        )
    else:
        # For text-to-video, a manifest may also provide the benchmark's exact
        # filename and clip length (e.g. VBench 1.0 prompt-index.mp4).
        prompt = batch['prompts'][0]
        output_name = (batch['output_name'][0] if args.wbench_manifest
                       else f'{prompt[:100]}.mp4')
        output_path = os.path.join(args.output_folder, output_name)
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None
        sample_frames = (int(batch['num_output_frames'].item())
                         if args.wbench_manifest else args.num_output_frames)
        sampled_noise = torch.randn(
            [1, sample_frames,
             config.image_or_video_shape[2],
             config.image_or_video_shape[3],
             config.image_or_video_shape[4]],
            device=device, dtype=torch.bfloat16,
        )

    # Parse camera trajectory if provided
    viewmats = None
    Ks = None
    traj_str = None
    row_event = None
    row_event_start = None
    if args.wbench_manifest:
        import numpy as np
        ev = batch.get("event_prompt", [""])[0]
        row_event = ev if ev else None
        if row_event is not None:
            esf = batch.get("event_start_frame", [0])[0]
            row_event_start = int(esf.item() if hasattr(esf, "item") else esf)
        camera_path = batch.get('camera_path', [''])[0]
        if camera_path:
            camera = np.load(camera_path)
            viewmats = torch.from_numpy(camera['viewmats']).unsqueeze(0).to(
                device=device, dtype=torch.bfloat16)
            Ks = torch.from_numpy(camera['Ks']).unsqueeze(0).to(
                device=device, dtype=torch.bfloat16)
    elif trajectory_list:
        traj_str = trajectory_list[idx]
    elif args.trajectory:
        traj_str = args.trajectory
    if traj_str:
        import numpy as np
        viewmats_np = parse_trajectory(traj_str)
        if args.trajectory_speed != 1.0:
            if args.trajectory_speed <= 0:
                raise ValueError("--trajectory_speed must be positive")
            # Scale the physical camera-centre displacement, not the PRoPE
            # residual.  For w/a/s/d/u/dn this changes motion speed while
            # preserving orientation and the first-frame gauge.
            c2w = np.linalg.inv(viewmats_np)
            origin = c2w[:1, :3, 3].copy()
            c2w[:, :3, 3] = origin + args.trajectory_speed * (
                c2w[:, :3, 3] - origin)
            viewmats_np = np.linalg.inv(c2w).astype(np.float32)
        # Default intrinsics (normalized)
        fx, fy, cx, cy = 0.5050505, 0.89786756, 0.5, 0.5
        Ks_np = np.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]] * len(viewmats_np), dtype=np.float32)
        viewmats = torch.from_numpy(viewmats_np).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        Ks = torch.from_numpy(Ks_np).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    actions_t = None
    embodiment_t = None
    if args.action_tokens:
        import numpy as np
        import re as _re
        from utils.action_encoding import (
            from_tokens, KEYBOARD, WBENCH_ACTION)
        toks = []
        for seg in args.action_tokens.split(","):
            seg = seg.strip()
            if not seg:
                continue
            m = _re.fullmatch(r"([A-Za-z+]+)(?:\*(\d+))?", seg)
            if m is None:
                raise ValueError(f"cannot parse action segment '{seg}'")
            toks += [m.group(1)] * int(m.group(2) or 1)
        arr = from_tokens(
            toks, table=KEYBOARD if args.action_keys else WBENCH_ACTION,
            magnitude=args.action_scale)
        actions_t = torch.from_numpy(arr).unsqueeze(0).to(
            device=device, dtype=torch.bfloat16)
        if args.embodiment_id is not None:
            embodiment_t = torch.tensor([args.embodiment_id], device=device).long()

    if args.reverse_time_attention:
        if initial_latent is not None:
            raise ValueError("reverse-time attention probe is T2V-only; a first-frame I2V condition would become a final-frame condition")
        sampled_noise = sampled_noise.flip(1)
        if viewmats is not None:
            viewmats = viewmats.flip(1)
            Ks = Ks.flip(1)

    # Generate frames. For clips longer than the model's trained temporal
    # horizon, periodically reset KV and continue from the previous window's
    # final clean latent. This bounds autoregressive cache error while keeping
    # semantic/visual continuity through an actual latent reference.
    if args.window_latent_frames > 0:
        if not args.i2v or initial_latent is None:
            raise ValueError("--window_latent_frames currently requires I2V")
        window_total = int(args.window_latent_frames)
        if window_total <= 1 or window_total % int(config.num_frame_per_block):
            raise ValueError("window latent frames must be a block-aligned value > 1")
        generated_first = window_total - 1
        overlap = int(args.window_overlap_latents)
        if overlap <= 0 or overlap % int(config.num_frame_per_block):
            raise ValueError("window overlap must be a positive block-aligned value")
        cursor = 0
        global_latents = 0
        window_ref = initial_latent
        window_videos = []
        window_latents = []
        original_sequence_ref = bool(getattr(
            pipeline, "ref_sequence_conditioning", False))
        target_total_latents = 1 + sampled_noise.shape[1]
        window_index = 0
        while global_latents < target_total_latents:
            if window_index == 0:
                generated_now = min(generated_first, sampled_noise.shape[1] - cursor)
                context_now = 1
                pipeline.ref_sequence_conditioning = original_sequence_ref
                window_start = 0
            else:
                remaining = target_total_latents - global_latents
                generated_now = min(window_total - overlap, remaining)
                generated_now -= generated_now % int(config.num_frame_per_block)
                if generated_now <= 0:
                    break
                context_now = overlap
                pipeline.ref_sequence_conditioning = False
                window_ref = window_latents[-1][:, -overlap:].detach()
                window_start = global_latents - overlap
            # A reset cache has local token coordinates starting at zero.
            # Preserve global camera motion by slicing the corresponding
            # trajectory window, rather than passing a global token offset to
            # an empty cache (which leaves no writable local cache span).
            total_now = context_now + generated_now
            vm_window = (viewmats[:, window_start:window_start + total_now]
                         if viewmats is not None else None)
            ks_window = (Ks[:, window_start:window_start + total_now]
                         if Ks is not None else None)
            local_event_start = max(
                0, int(row_event_start if row_event_start is not None
                       else args.event_start_frame) - window_start)
            import inspect as _inspect
            _window_params = _inspect.signature(pipeline.inference).parameters
            win_video, win_latents = pipeline.inference(
                noise=sampled_noise[:, cursor:cursor + generated_now],
                text_prompts=prompts,
                return_latents=True,
                initial_latent=window_ref,
                viewmats=vm_window,
                Ks=ks_window,
                event_prompts=(row_event or args.event_prompt),
                event_start_frame=local_event_start,
                **({"start_frame_index": 0}
                   if "start_frame_index" in _window_params else {}),
                **({"concat_initial_latent": True}
                   if "concat_initial_latent" in _window_params else {}),
            )
            if not window_videos:
                window_videos.append(win_video)
                window_latents.append(win_latents)
            else:
                # Wan's temporal VAE maps N overlapping latents to 4N-3 RGB
                # frames. These are the same times in both windows, so this is
                # a true overlap-add rather than a morph between unrelated
                # frames.
                overlap_rgb = 4 * overlap - 3
                if args.window_color_match:
                    match = min(overlap_rgb, window_videos[-1].shape[1],
                                win_video.shape[1])
                    prev_overlap = window_videos[-1][:, -match:].float()
                    next_overlap = win_video[:, :match].float()
                    dims = (1, 3, 4)
                    prev_mean = prev_overlap.mean(dim=dims, keepdim=True)
                    next_mean = next_overlap.mean(dim=dims, keepdim=True)
                    prev_std = prev_overlap.std(dim=dims, keepdim=True)
                    next_std = next_overlap.std(dim=dims, keepdim=True)
                    raw_gain = prev_std / next_std.clamp_min(1e-4)
                    raw_bias = prev_mean - raw_gain * next_mean
                    strength = float(args.window_color_match_strength)
                    gain = (1.0 + strength * (raw_gain - 1.0)).clamp(0.95, 1.05)
                    bias = (strength * raw_bias).clamp(-0.02, 0.02)
                    win_video = (gain.to(win_video.dtype) * win_video
                                 + bias.to(win_video.dtype)).clamp(0, 1)
                    print(f"[window] overlap color match gain="
                          f"{gain.flatten().tolist()} bias={bias.flatten().tolist()}")
                requested_fade = int(args.window_crossfade_rgb)
                fade = min(overlap_rgb,
                           requested_fade if requested_fade > 0 else overlap_rgb,
                           window_videos[-1].shape[1], win_video.shape[1])
                if fade > 0:
                    alpha = torch.linspace(
                        0.0, 1.0, fade + 2, device=win_video.device,
                        dtype=win_video.dtype)[1:-1].view(1, fade, 1, 1, 1)
                    window_videos[-1][:, -fade:] = (
                        (1 - alpha) * window_videos[-1][:, -fade:]
                        + alpha * win_video[:, :fade])
                    # The complete overlap denotes the same timestamps in
                    # both decoded windows.  Cross-fading only `fade` frames
                    # does not make the remaining overlap new footage: keeping
                    # it duplicated 4*overlap-3-fade RGB frames at every reset
                    # (9 frames for overlap=4, fade=4), causing a periodic
                    # pause/flash and an increasingly wrong final timestamp.
                    window_videos.append(win_video[:, overlap_rgb:])
                else:
                    window_videos.append(win_video[:, overlap_rgb:])
                window_latents.append(win_latents[:, overlap:])
            cursor += generated_now
            global_latents += generated_now + (1 if window_index == 0 else 0)
            window_index += 1
        pipeline.ref_sequence_conditioning = original_sequence_ref
        if not window_videos:
            raise ValueError("requested output is shorter than one long-video window")
        if cursor < sampled_noise.shape[1]:
            print(f"[window] dropping {sampled_noise.shape[1] - cursor} trailing "
                  "noise latents to keep reference+window block alignment")
        video = torch.cat(window_videos, dim=1)
        latents = torch.cat(window_latents, dim=1)
        print(f"[window] generated {len(window_videos)} overlapping KV-reset windows, "
              f"video_frames={video.shape[1]}, latent_frames={latents.shape[1]}")
    else:
        import inspect as _inspect
        _inference_params = _inspect.signature(pipeline.inference).parameters
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            viewmats=viewmats,
            Ks=Ks,
            event_prompts=(row_event or args.event_prompt),
            event_start_frame=(row_event_start if row_event_start is not None
                               else args.event_start_frame),
            # Same reason as `profile` below: the bidirectional pipelines take
            # no action arguments at all, and passing them unconditionally
            # killed every non-causal run with a TypeError before it generated
            # anything. Ask the pipeline what it accepts instead of assuming.
            **({k: v for k, v in (("actions", actions_t),
                                  ("embodiment_id", embodiment_t))
                if k in _inference_params and v is not None}),
            # Only CausalInferencePipeline measures the init/diffusion/VAE split;
            # the others take no `profile`, and passing it unconditionally made
            # every non-causal run die with a TypeError before generating anything.
            **({"profile": True} if args.profile else {}),
        )
    if os.environ.get("WAN_SKIP_DECODE") == "1":
        latent_path = output_path.replace(".mp4", "_latents.pt")
        torch.save(latents[0].detach().cpu(), latent_path)
        print(f"[latent-only] saved {latent_path}", flush=True)
        continue
    if args.reverse_time_attention:
        video = video.flip(1)
        latents = latents.flip(1)

    if args.camera_diag and local_rank == 0:
        print("[camera-diag] causal PRoPE contribution by transformer block")
        for block_idx, block in enumerate(pipeline.generator.model.blocks):
            diag = getattr(block.self_attn, "_camera_diag", {})
            for token_start, values in sorted(diag.items()):
                print(f"[camera-diag] block={block_idx} token_start={token_start} "
                      f"base={values['base_rms']:.6g} "
                      f"attn={values['prope_attention_rms']:.6g} "
                      f"projected={values['prope_projected_rms']:.6g} "
                      f"ratio={values['projected_to_base']:.6g}")
        if args.camera_diag_path:
            diag_payload = {
                block_idx: getattr(block.self_attn, "_camera_diag", {})
                for block_idx, block in enumerate(pipeline.generator.model.blocks)
            }
            os.makedirs(os.path.dirname(args.camera_diag_path) or ".", exist_ok=True)
            torch.save(diag_payload, args.camera_diag_path)
            print(f"[camera-diag] saved {args.camera_diag_path}")

    # Record latency on rank 0; first prompt is warmup → None.
    # All pipelines stop the timer before VAE decode (see pipeline.last_chunk0_latency).
    if local_rank == 0:
        sample_lat = getattr(pipeline, "last_chunk0_latency", None)
        if len(chunk0_latencies) >= 1:
            chunk0_latencies.append(sample_lat)
        else:
            chunk0_latencies.append(None)

    # A causal I2V model has an unavoidable cold-start asymmetry: the first
    # decoded frames are driven almost entirely by the single reference image,
    # before generated history exists.  On the LongLive checkpoint that prior
    # is a short zoom-in, strong enough to make an S command look like W for
    # roughly half a second even though every later block is correctly signed.
    # Apply the commanded camera scale during that cold-start interval and keep
    # the attained scale afterwards.  This is a camera transform (not frame
    # reversal), and is deliberately limited to pure forward/backward openings.
    if bool(config.get("opening_direction_correction", False)) and traj_str:
        import re
        match = re.match(r"\s*(w|s)(?:\*|,|$)", traj_str.lower())
        if match:
            direction = match.group(1)
            opening_frames = max(2, int(config.get(
                "opening_direction_frames", 12)))
            strength_key = ("opening_forward_strength" if direction == "w"
                            else "opening_backward_strength")
            strength = float(config.get(
                strength_key, 0.06 if direction == "w" else 0.12))
            t_count = video.shape[1]
            ramp = torch.arange(
                t_count, device=video.device, dtype=torch.float32
            ).div(float(opening_frames)).clamp_(0, 1)
            # Forward affine scale: >1 zooms in, <1 zooms out. grid_sample
            # consumes the inverse mapping, hence theta uses 1/scale.
            signed = strength if direction == "w" else -strength
            scale = 1.0 + signed * ramp
            theta = torch.zeros(
                t_count, 2, 3, device=video.device, dtype=torch.float32)
            theta[:, 0, 0] = 1.0 / scale
            theta[:, 1, 1] = 1.0 / scale
            frames = video[0].float()
            grid = F.affine_grid(theta, frames.shape, align_corners=False)
            corrected = F.grid_sample(
                frames, grid, mode="bicubic", padding_mode="border",
                align_corners=False)
            video = video.clone()
            video[0] = corrected.to(video.dtype).clamp_(0, 1)
            print(f"[camera-opening] {direction=} frames={opening_frames} "
                  f"strength={strength:.3f}", flush=True)

    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu() 
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    traj_suffix = "_" + traj_str.replace("*", "").replace(",", "") if traj_str else ""
    if args.reverse_time_attention:
        traj_suffix += "_future_only"
    if args.wbench_manifest:
        output_path = os.path.join(args.output_folder, batch['output_name'][0])
    else:
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}{traj_suffix}.mp4')
    if args.save_latents and not (args.sp_size > 1 and local_rank != 0):
        torch.save(clean_latent, output_path.replace(".mp4", "_latents.pt"))
    if not (args.sp_size > 1 and local_rank != 0):
        writer = imageio.get_writer(output_path, fps=args.fps)
        for frame in video[0]:
            writer.append_data(frame.numpy().astype('uint8'))
        writer.close()
    if dist.is_initialized():
        dist.barrier()


# Aggregate latency on rank 0 (drop the first prompt's warmup).
if local_rank == 0:
    valid = [v for v in chunk0_latencies[1:] if v is not None]
    if valid:
        print(f"[timing] rank0 chunk0 latency excl. decode (from 2nd prompt): "
              f"avg={sum(valid)/len(valid):.3f}s over {len(valid)} samples")

if local_rank == 0 and torch.cuda.is_available():
    print(f"[vram] peak allocated {torch.cuda.max_memory_allocated(gpu) / 1024**3:.2f} GB, "
          f"peak reserved {torch.cuda.max_memory_reserved(gpu) / 1024**3:.2f} GB")
