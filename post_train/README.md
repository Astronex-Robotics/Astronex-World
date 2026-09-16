# Post-training

```bash
python post_train/train.py --recipe sft --gpus 2      # supervised fine-tuning
python post_train/train.py --recipe dmd --gpus 2      # distribution matching
```

Both start from the released weights with a freshly initialised adapter, so
step 0 reproduces the release exactly — `lora_b` is zero, which makes the
adapter an identity at the start.

## Which recipe

| | sft | dmd |
|---|---|---|
| trainer | `camera_diffusion` | `camera_score_distillation` |
| objective | flow matching on the dataset's latents | `fake_score − real_score` |
| networks on the card | student | student + one shared score backbone |
| few-step sampling | no | yes, this is what buys it |
| ball bounce reached | 36.1 px (41% of teacher) | 16.6–21.2 px (19–24%) |
| lr | 2e-6, action 2e-5 | 1e-6, critic 1e-6 |

**SFT is the better tool for picture quality and for a new corpus**, and it is
cheaper and steadier. It is also, measurably, much better at physics — 41% of
the teacher against DMD's 19–24%. What it cannot do is make the model samplable
in eight steps; that is what DMD is for, and the physics loss is the price.

## What is in the DMD objective

Four terms, each answering something an earlier round got wrong:

- **DMD** — the distribution supervision. Teacher score fixed, critic score
  tracking the student, gradient from the difference. Not a per-pixel loss: a
  sample can be right in distribution and still look noisy, which is why the
  other three exist.
- **`motion_preserve_weight: 2.0`** — matches the student's mean temporal
  difference to the clip's own. Worth 10% → 19% of teacher bounce on its own.
  The target comes from the batch, so it asks for the motion the clip actually
  has rather than a fixed amount.
- **`rolling_forcing_prob: 1.0`** — staggers the noise level across the window
  so later frames denoise from noisier states, which is the regime a long
  rollout actually runs in. Every step, not half of them: drift shows up in the
  tail, which is exactly what the ramp trains against.
- **GAN** — `gan_weight: 0.05`, a 14M pooled discriminator head on the critic
  reading blocks 13/21/29. Against the high-frequency noise the DMD term
  tolerates. `gan_max_timestep: 250` keeps it near the clean end: at high noise
  the injected noise dominates and the discriminator would be judging what it
  was handed rather than the sample's own texture.

### Reading the GAN in the log

```
gan_d_loss=1.221  d_real=0.072  d_fake=-0.279     healthy
gan_d_loss=1.386  d_real=0.005  d_fake=0.004      learning nothing
```

1.386 is 2·ln 2 — chance exactly. Pinned there with `d_real` and `d_fake` equal
means the discriminator cannot separate real from generated, and the
generator's adversarial term is spending two forward passes a step pushing
against nothing. That happened here from applying `lr_critic` (1e-6, sized for
adapters continuing a long fine-tune) to a randomly initialised head, which is
why the head has its own `gan_lr: 3e-5`.

If the discriminator instead runs away — `d_loss` collapsing toward zero —
lower `gan_weight` before touching anything else.

## Distributed

NCCL is the backend. Single node needs nothing:

```bash
python post_train/train.py --recipe dmd --gpus 2
```

Across hosts, each node runs the same command with its own rank:

```bash
# node 0
python post_train/train.py --recipe dmd --gpus 8 \
  --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 --port 30401
# node 1
python post_train/train.py --recipe dmd --gpus 8 \
  --nnodes 2 --node-rank 1 --master-addr 10.0.0.1 --port 30401
```

`--nnodes > 1` sets `NCCL_SOCKET_IFNAME=^lo,docker` unless it is already in the
environment: single-node NCCL finds its transport on its own, but across hosts
it picks an interface and on a box with several it can pick one the peers
cannot reach. `--nccl-debug` turns on `NCCL_DEBUG=INFO`, which is what to reach
for when a multi-node job hangs on its first collective rather than failing.

`--sp-size N` switches to the sequence-parallel init and shards the sequence
across N ranks. Useful for longer windows than one card's attention can hold;
it changes the data sampler, so a run cannot be resumed across a change in it.

## Memory

Two 44 GB cards: ~15 GiB resident, ~39 GiB peak in the critic phase for DMD.
Three settings buy that room and all three are on in `configs/dmd.yaml`:

- **`share_score_backbone`** — teacher and critic are two adapters over one
  network, the teacher reading it with the adapter switched off. `real_ckpt`
  and `fake_ckpt` already name the same file, and the critic is re-initialised
  from `fake_ckpt` on every launch (`save()` writes only the generator), so
  nothing is lost. The second 5.36B backbone was duplication.
- **`split_dtype_fsdp`** — frozen backbone in bf16, fp32 adapters handed to
  FSDP as `ignored_states` so every flattened unit stays dtype-uniform.
  Numerically a no-op for the compute, which already ran bf16 under
  `mixed_precision`; it only stops keeping fp32 masters for parameters that
  never update. Their gradients are all-reduced by hand, since FSDP reduces
  only what it shards.
- **`text_encoder_cpu_offload`** — UMT5 is frozen and embeds one short prompt a
  step. 10.6 GiB of resident VRAM for that is the least useful allocation on
  the card.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the launcher.
Fragmentation, not capacity, is what this run kept dying on: 84 MiB short with
745 MiB reserved and unallocated.

`--memlog` prints resident and peak memory per phase, which is the first thing
to look at on different hardware.

## Two silent traps

Both were hit here. Both produce a run that looks fine.

**Non-reentrant checkpointing pins the KV cache.** `use_reentrant=False` wraps
each block in `saved_tensors_hooks`, and the pack/unpack closures hold a
`_CheckpointFrame` holding that block's forward kwargs — the KV cache among
them. Thirty frames survived every step, one per block, retaining 5.6 GiB each
time; the run went out of memory on the third step regardless of how much was
moved off the card. The fix in this tree passes those kwargs on the module
instead of through the checkpoint call, so the frames pin only the module, and
empties the attribute once the step's backward is done.

**Reentrant checkpointing is not the alternative.** `use_reentrant=True` only
runs the checkpointed region's backward when one of its positional inputs
requires grad. Block 0's input comes off a frozen patch embedding, so it does
not — and the whole 30-block chain silently builds no graph: 600 trainable
tensors, all 600 with `grad is None`, the optimiser moving nothing, and 26
steps of checkpoints byte-identical to their starting point. It reads as a
memory fix, because a run computing no gradients has very little to hold.

**So: if a run's checkpoints come out equal to where they started, compare them
tensor by tensor before believing anything measured from them.**

```python
import torch
a = torch.load("start/model.pt")["generator_ema"]
b = torch.load("end/model.pt")["generator_ema"]
print(sum(1 for k in a if not torch.equal(a[k], b[k])), "of", len(a), "changed")
```

## Data

`data_path: ./dataset/sft_phys` — LMDB shards of latents with camera poses,
actions and embodiment ids.

One caution worth more than it looks: a corpus whose `camera_extrinsics` are
identity trains PRoPE against a constant. Five separate interventions here
failed before that was noticed, because nothing about the loss says so. Check
that the poses in a new corpus actually vary over time before drawing any
conclusion from a run on it.
