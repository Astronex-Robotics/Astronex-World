"""Small CPU error buffer for LongLive/SVI-style history recycling."""

import random

import torch


class PositionErrorBuffer:
    def __init__(self, num_positions, num_buckets=50, size_per_bucket=32,
                 num_timesteps=1000, modulation=0.3):
        self.num_positions = int(num_positions)
        self.num_buckets = int(num_buckets)
        self.size_per_bucket = int(size_per_bucket)
        self.num_timesteps = int(num_timesteps)
        self.modulation = float(modulation)
        self.data = {(p, t): [] for p in range(self.num_positions)
                     for t in range(self.num_buckets)}
        self.total_added = 0

    def _bucket(self, timestep_index):
        value = int(timestep_index)
        return min(self.num_buckets - 1, max(
            0, value * self.num_buckets // self.num_timesteps))

    def add(self, position, timestep_index, error):
        key = (min(int(position), self.num_positions - 1),
               self._bucket(timestep_index))
        item = error.detach().to("cpu", dtype=torch.bfloat16, copy=True)
        bucket = self.data[key]
        if len(bucket) < self.size_per_bucket:
            bucket.append(item)
        else:
            bucket[random.randrange(self.size_per_bucket)] = item
        self.total_added += 1

    def sample(self, position, timestep_index, device, dtype):
        p = min(int(position), self.num_positions - 1)
        t = self._bucket(timestep_index)
        # Nearby timestep buckets make the buffer useful during warmup.
        choices = []
        for radius in range(self.num_buckets):
            for candidate in (t - radius, t + radius):
                if 0 <= candidate < self.num_buckets:
                    choices.extend(self.data[(p, candidate)])
            if choices:
                break
        if not choices:
            return None
        scale = random.uniform(1.0 - self.modulation, 1.0 + self.modulation)
        return random.choice(choices).to(device=device, dtype=dtype) * scale

    def sample_position(self, position, device, dtype):
        """Sample an error at a sequence position, independent of timestep.

        LongLive uses this path for E_img because a clean KV prefix is the
        result of a complete rollout; its error is position-dependent but no
        longer associated with the timestep currently sampled for the target.
        """
        p = min(int(position), self.num_positions - 1)
        choices = []
        for t in range(self.num_buckets):
            choices.extend(self.data[(p, t)])
        if not choices:
            return None
        scale = random.uniform(1.0 - self.modulation, 1.0 + self.modulation)
        return random.choice(choices).to(device=device, dtype=dtype) * scale
