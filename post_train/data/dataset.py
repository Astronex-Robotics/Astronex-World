from post_train.data.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


# Readers reserve a large virtual map so they tolerate an LMDB that grows after
# they opened it. Without this, reading a dataset while its builder is still
# writing dies with MDB_MAP_RESIZED partway through the first epoch. The value is
# address space, not memory or disk -- nothing is allocated by asking for it.
READER_MAP_SIZE = 1 << 40  # 1 TiB






class LatentLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True, map_size=READER_MAP_SIZE,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1]
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }



class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }





class CameraLatentLMDBDataset(LatentLMDBDataset):
    """LatentLMDBDataset extended with per-frame camera data for PRoPE.

    Expects the LMDB to contain raw camera parameters:
      - ``intrinsics``: float32 array of shape ``(N, 4)`` — [fx, fy, cx, cy] normalized
      - ``poses``:      float32 array of shape ``(N, F, 7)`` — [tx,ty,tz, qx,qy,qz,qw] w2c

    viewmats (F, 4, 4) and Ks (F, 3, 3) are built on-the-fly via
    ``build_viewmats_and_Ks()``, which normalizes poses to the first frame.

    ``data_path`` can be either:
      - a single LMDB directory (has ``data.mdb`` inside), or
      - a parent directory containing multiple LMDB subdirectories (sharding).
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        # Detect sharding: if data_path contains data.mdb, it's a single LMDB;
        # otherwise treat each subdirectory as a shard.
        if os.path.isfile(os.path.join(data_path, "data.mdb")):
            self._sharded = False
            super().__init__(data_path, max_pair)
            self.intrinsics_shape = get_array_shape_from_lmdb(
                self.env, 'intrinsics')
            self.poses_shape = get_array_shape_from_lmdb(self.env, 'poses')
        else:
            self._sharded = True
            self.envs = []
            self.index = []  # list of (shard_id, local_idx)
            self._latents_shapes = []
            self._intrinsics_shapes = []
            self._poses_shapes = []
            for fname in sorted(os.listdir(data_path)):
                sub = os.path.join(data_path, fname)
                if not os.path.isdir(sub):
                    continue
                if not os.path.isfile(os.path.join(sub, "data.mdb")):
                    continue
                env = lmdb.open(sub, readonly=True, lock=False,
                                map_size=READER_MAP_SIZE,
                                readahead=False, meminit=False)
                sid = len(self.envs)
                self.envs.append(env)
                ls = get_array_shape_from_lmdb(env, 'latents')
                self._latents_shapes.append(ls)
                self._intrinsics_shapes.append(
                    get_array_shape_from_lmdb(env, 'intrinsics'))
                self._poses_shapes.append(
                    get_array_shape_from_lmdb(env, 'poses'))
                for j in range(ls[0]):
                    self.index.append((sid, j))
            self.max_pair = max_pair

    def __len__(self):
        if self._sharded:
            return min(len(self.index), self.max_pair)
        return super().__len__()

    def __getitem__(self, idx):
        if self._sharded:
            sid, local_idx = self.index[idx]
            env = self.envs[sid]
            ls = self._latents_shapes[sid]
            latents = retrieve_row_from_lmdb(
                env, "latents", np.float16, local_idx, shape=ls[1:])
            if len(latents.shape) == 4:
                latents = latents[None, ...]
            prompts = retrieve_row_from_lmdb(env, "prompts", str, local_idx)
            intrinsics = retrieve_row_from_lmdb(
                env, "intrinsics", np.float32, local_idx,
                shape=self._intrinsics_shapes[sid][1:])
            poses = retrieve_row_from_lmdb(
                env, "poses", np.float32, local_idx,
                shape=self._poses_shapes[sid][1:])
        else:
            # Single LMDB path — original behavior
            latents = retrieve_row_from_lmdb(
                self.env, "latents", np.float16, idx,
                shape=self.latents_shape[1:])
            if len(latents.shape) == 4:
                latents = latents[None, ...]
            prompts = retrieve_row_from_lmdb(self.env, "prompts", str, idx)
            intrinsics = retrieve_row_from_lmdb(
                self.env, "intrinsics", np.float32, idx,
                shape=self.intrinsics_shape[1:])
            poses = retrieve_row_from_lmdb(
                self.env, "poses", np.float32, idx,
                shape=self.poses_shape[1:])

        viewmats, Ks = build_viewmats_and_Ks(intrinsics, poses)
        out = {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1],
            "viewmats": torch.tensor(viewmats, dtype=torch.float32),
            "Ks": torch.tensor(Ks, dtype=torch.float32),
        }
        # Control2V stores poses and no actions; CrossFPS stores actions and
        # all-identity poses. They are the same control signal split across two
        # datasets, which is why the camera branch was trained on motion with no
        # command and the action branch on commands with no motion. Decomposed
        # into the previous camera's frame these poses quantise to exactly the
        # +-0.08 / +-3 degrees of camera_trajectory._MOTIONS, so the WASD that
        # generated them is recovered rather than estimated.
        if bool(getattr(self, "actions_from_poses", False)):
            from utils.action_encoding import from_viewmats
            out["actions"] = torch.tensor(
                from_viewmats(viewmats), dtype=torch.float32)
            out["embodiment_id"] = torch.zeros((), dtype=torch.long)
        return out


def build_viewmats_and_Ks(intrinsics, poses):
    """Build 4x4 w2c view matrices and 3x3 intrinsics from raw poses.

    Called at dataset load time (in ``CameraLatentLMDBDataset.__getitem__``).

    Args:
        intrinsics: (4,) ndarray [fx, fy, cx, cy] (normalized)
        poses:      (T, 7) ndarray [tx, ty, tz, qx, qy, qz, qw] w2c OpenCV

    Returns:
        viewmats: (T, 4, 4) float32 — w2c SE3, normalized to first frame
        Ks:       (T, 3, 3) float32 — intrinsics
    """
    T = len(poses)
    fx, fy, cx, cy = intrinsics

    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(T):
        tx, ty, tz, qx, qy, qz, qw = poses[i]
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        viewmats[i, :3, :3] = R
        viewmats[i, :3, 3] = [tx, ty, tz]
        viewmats[i, 3, 3] = 1.0

    # Normalize: align all poses to first frame
    c2w = np.linalg.inv(viewmats)
    C0_inv = np.linalg.inv(c2w[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w])
    viewmats = np.linalg.inv(c2w_aligned).astype(np.float32)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    Ks = np.tile(K, (T, 1, 1))

    return viewmats, Ks


def cycle(dl):
    while True:
        for data in dl:
            yield data


# ---------------------------------------------------------------------------
# Action-conditioned variants
# ---------------------------------------------------------------------------

def optional_array_shape(env, array_name):
    """Shape of an LMDB array, or None when the key is absent.

    Action labels are optional: the same camera LMDB should be usable for a
    camera-only stage and for an action-conditioned one, and a dataset built
    before actions existed must keep loading rather than crashing.
    """
    try:
        with env.begin() as txn:
            raw = txn.get(f"{array_name}_shape".encode())
    except Exception:
        return None
    if raw is None:
        return None
    return tuple(map(int, raw.decode().split()))


def _read_event_prompt(env, idx):
    """Optional per-row event prompt.

    Stored as its own ``event_prompts`` column rather than appended to the
    caption, so a dataset can be re-labelled with events without rewriting the
    captions and so the two channels stay independently droppable at train
    time. Absent for every dataset built before events existed, which is why
    a missing column reads as empty rather than raising.

    Always returns the key, empty string included. A batch that mixes rows
    with and without it would otherwise be a batch of dicts with different key
    sets, which the default collate refuses -- and mixing sources is exactly
    what this stage does.
    """
    try:
        text = retrieve_row_from_lmdb(env, "event_prompts", str, idx)
    except Exception:
        text = ""
    return {"event_prompts": text or ""}


def _read_actions(env, idx, actions_shape, embodiment_shape):
    """Per-frame action vector and embodiment id for one row.

    LMDB keys:
      - ``actions``:    float32 (N, F, action_dim) — Cosmos-compatible, 64-wide
      - ``embodiment``: int64   (N,)               — embodiment domain id

    Returns an empty dict when the LMDB carries no actions, so callers can
    ``update()`` unconditionally.
    """
    out = {}
    if actions_shape is not None:
        actions = retrieve_row_from_lmdb(
            env, "actions", np.float32, idx, shape=actions_shape[1:])
        out["actions"] = torch.tensor(np.ascontiguousarray(actions), dtype=torch.float32)
    if embodiment_shape is not None:
        emb = retrieve_row_from_lmdb(
            env, "embodiment", np.int64, idx, shape=embodiment_shape[1:])
        out["embodiment_id"] = torch.tensor(np.asarray(emb).reshape(-1)[0], dtype=torch.long)
    return out


class ActionCameraLatentLMDBDataset(CameraLatentLMDBDataset):
    """:class:`CameraLatentLMDBDataset` plus optional per-frame action labels.

    Camera and action are stored and returned side by side because they are
    genuinely independent control channels -- a driving clip has both, a
    handheld-video clip has only camera, a fixed-camera manipulation clip has
    only actions. Making actions optional at the row level keeps one dataset
    class serving all three.
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        super().__init__(data_path, max_pair)
        if self._sharded:
            self._actions_shapes = [optional_array_shape(e, "actions") for e in self.envs]
            self._embodiment_shapes = [optional_array_shape(e, "embodiment") for e in self.envs]
        else:
            self._actions_shape = optional_array_shape(self.env, "actions")
            self._embodiment_shape = optional_array_shape(self.env, "embodiment")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self._sharded:
            sid, local_idx = self.index[idx]
            item.update(_read_actions(
                self.envs[sid], local_idx,
                self._actions_shapes[sid], self._embodiment_shapes[sid]))
            item.update(_read_event_prompt(self.envs[sid], local_idx))
        else:
            item.update(_read_actions(
                self.env, idx, self._actions_shape, self._embodiment_shape))
            item.update(_read_event_prompt(self.env, idx))
        return item




