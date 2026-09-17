"""Flow matching core algorithms.

Pure tensor operations shared by the Astronex trainers.
No external dependencies beyond torch.
"""

import torch
from torch import Tensor
from typing import Optional


def add_flow_noise(clean: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    """Construct noisy sample via flow matching interpolation.

    x_t = (1 - sigma) * x_0 + sigma * epsilon

    Args:
        clean: clean sample x_0, any shape
        noise: noise epsilon, same shape as clean
        sigma: noise level, broadcastable to clean shape
    """
    return (1 - sigma) * clean + sigma * noise


def flow_matching_target(clean: Tensor, noise: Tensor) -> Tensor:
    """Compute velocity target for flow matching.

    v = epsilon - x_0
    """
    return noise - clean


def pred_x0_from_flow(
    noisy: Tensor,
    flow_pred: Tensor,
    sigma: Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Predict clean sample from flow prediction.

    x_0 = x_t - sigma * v

    Args:
        noisy: noisy input x_t
        flow_pred: predicted velocity v
        sigma: noise level, broadcastable
        compute_dtype: dtype for the computation (default fp32 for stability)
    """
    original_dtype = noisy.dtype
    return (
        noisy.to(compute_dtype) - sigma.to(compute_dtype) * flow_pred.to(compute_dtype)
    ).to(original_dtype)


def flow_matching_loss(
    pred: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Weighted MSE loss for flow matching.

    Args:
        pred: model prediction
        target: ground truth target (velocity or x0)
        weight: per-element or per-sample weight, broadcastable
        mask: boolean mask, True = include in loss
        reduction: "mean" or "none"
    """
    loss = (pred.float() - target.float()).pow(2)
    if weight is not None:
        loss = loss * weight
    if mask is not None:
        loss = loss[mask]
    if reduction == "mean":
        return loss.mean()
    return loss
