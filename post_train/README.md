# Post-training

```bash
bash scripts/post_train_camera.sh --data <lmdb dir>   # camera only
bash scripts/post_train_action.sh --data <lmdb dir>   # action in and out only
bash scripts/post_train_sft.sh    --data <lmdb dir>   # both together
```

All three run the `camera_diffusion` trainer: flow matching on the dataset's
latents, teacher-forced. All start from the released weights with a freshly
initialised adapter, so step 0 reproduces the release exactly — `lora_b` is
zero, which makes the adapter an identity at the start — and the backbone
stays frozen.

## Which recipe

| | camera | action | sft |
|---|---|---|---|
| camera (PRoPE) | trains | — | trains |
| action input | — | trains | trains |
| action head | — | trains | — |
| lr | 2e-6, camera 1e-6 | 2e-6, action 2e-5 | 2e-6, camera 1e-6, action 2e-5 |

**camera** and **action** each move one pathway and leave the other where the
release left it. That is what makes a question like "did this corpus change
camera control?" answerable: the pathway that was not trained is bit-identical,
not merely similar.

**sft** trains both together and refines control on a new corpus; it reached
36.1 px of ball bounce against the teacher's 88.8, the best physics of any
student here.

The action head is trained only by `action`, through `_action_output_loss` in
`post_train/objectives/camera_diffusion.py`. The other two do not install it at
all (`action_output: false`): the head ships
zero-initialised, so frozen it returns zeros and its loss has no gradient to
give, and the six head tensors in the checkpoint are dropped on load with a log
line saying so.

## Distributed

NCCL is the backend. Single node needs nothing:

```bash
python post_train/train.py --recipe sft --gpus 2
```

Across hosts, each node runs the same command with its own rank:

```bash
# node 0
python post_train/train.py --recipe sft --gpus 8 \
  --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 --port 30401
# node 1
python post_train/train.py --recipe sft --gpus 8 \
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

Two 44 GB cards. A single 44 GB card runs out of memory in the first backward
pass: FSDP needs the second rank to shard the trainable state.

- **`text_encoder_cpu_offload`** — UMT5 is frozen and embeds one short prompt a
  step. 10.6 GiB of resident VRAM for that is the least useful allocation on
  the card.
- **`split_dtype_fsdp`** must stay off whenever a control graft trains: the
  grafts are plain `nn.Linear` with no dtype boundary of their own, and
  casting them to bf16 would round lr-sized updates away. `train.py` refuses
  the combination.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the launcher.
Fragmentation, not capacity, is what these runs kept dying on: 84 MiB short
with 745 MiB reserved and unallocated.

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

`data_path: ./dataset/sft_phys` (relative to this repo; `--data <dir>` overrides) — LMDB shards of latents with camera poses,
actions and embodiment ids.

One caution worth more than it looks: a corpus whose `camera_extrinsics` are
identity trains PRoPE against a constant. Five separate interventions here
failed before that was noticed, because nothing about the loss says so. Check
that the poses in a new corpus actually vary over time before drawing any
conclusion from a run on it.
