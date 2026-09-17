"""Lightweight Pluecker-ray AdaLN/FiLM conditioning for camera control."""

import torch


def camera_ray_tokens(viewmats, Ks, grid_sizes, dtype=None):
    """Return per-latent-patch ray features [B, F*H*W, 6].

    ``viewmats`` are world-to-camera transforms and ``Ks`` use normalized image
    coordinates.  Following LingBot-World, each token receives its camera
    origin and unit viewing direction; no camera token enters cross-attention.
    """
    if viewmats is None or Ks is None:
        return None
    outputs = []
    for batch_index, (frames, height, width) in enumerate(grid_sizes.tolist()):
        w2c = viewmats[batch_index, :frames].float()
        c2w = torch.linalg.inv(w2c)
        K_inv = torch.linalg.inv(Ks[batch_index, :frames].float())
        ys = (torch.arange(height, device=w2c.device, dtype=torch.float32) + 0.5) / height
        xs = (torch.arange(width, device=w2c.device, dtype=torch.float32) + 0.5) / width
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(1, -1, 3)
        camera_dirs = torch.einsum("fij,fpj->fpi", K_inv, pixels.expand(frames, -1, -1))
        world_dirs = torch.einsum("fij,fpj->fpi", c2w[:, :3, :3], camera_dirs)
        world_dirs = torch.nn.functional.normalize(world_dirs, dim=-1, eps=1e-6)
        origins = c2w[:, None, :3, 3].expand_as(world_dirs)
        outputs.append(torch.cat((origins, world_dirs), dim=-1).reshape(1, -1, 6))
    result = torch.cat(outputs, dim=0)
    return result.to(dtype=dtype or viewmats.dtype)
