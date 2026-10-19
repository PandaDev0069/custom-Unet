import torch
import torch.nn as nn
import torch.nn.functional as F


def instance_boundaries(instance_ids: torch.Tensor) -> torch.Tensor:
    """Create a boundary target from a [B, H, W] integer instance-ID map.

    0 = background; 1, 2, ... = distinct filaments.
    Only contacts between two positive, distinct IDs are marked. Background
    contours and ambiguous overlap IDs (-1) are excluded. The production path
    uses precomputed proximity targets from independent annotation masks instead.
    Returns [B, 1, H, W] of type float32.
    """
    if instance_ids.ndim != 3:
        raise ValueError("instance_ids must have shape [B, H, W]")

    boundary = torch.zeros_like(instance_ids, dtype=torch.bool)

    a, b = instance_ids[:, 1:, :], instance_ids[:, :-1, :]
    vertical = (a != b) & (a > 0) & (b > 0)
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical

    a, b = instance_ids[:, :, 1:], instance_ids[:, :, :-1]
    horizontal = (a != b) & (a > 0) & (b > 0)
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal

    return boundary.unsqueeze(1).float()


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation to morphological erosion."""
    a = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    b = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(a, b)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(soft_erode(x), kernel_size=3, stride=1, padding=1)


def soft_skeleton(x: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """Differentiable soft skeleton used by the soft-clDice surrogate."""
    skeleton = F.relu(x - soft_open(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    probabilities: torch.Tensor, targets: torch.Tensor, iterations: int = 5
) -> torch.Tensor:
    predicted_centerline = soft_skeleton(probabilities, iterations)
    true_centerline = soft_skeleton(targets, iterations)

    axes = (1, 2, 3)
    eps = 1e-6
    precision = (
        (predicted_centerline * targets).sum(dim=axes) + eps
    ) / (predicted_centerline.sum(dim=axes) + eps)
    sensitivity = (
        (true_centerline * probabilities).sum(dim=axes) + eps
    ) / (true_centerline.sum(dim=axes) + eps)

    return (1 - (2 * precision * sensitivity + eps) / (
        precision + sensitivity + eps
    )).mean()


class FilamentPQProxyLoss(nn.Module):
    """Differentiable *proxy* for PQ-oriented filament segmentation.

    Inputs
    ------
    logits : [B, 2, H, W]
        channel 0 = foreground logits
        channel 1 = instance-boundary logits
    instance_ids : [B, H, W]
        0 = background; positive integers identify distinct filaments

    This does NOT directly optimize official Panoptic Quality.
    """
    def __init__(
        self,
        bce_weight: float = 0.35,
        dice_weight: float = 0.40,
        boundary_weight: float = 0.25,
        cldice_weight: float = 0.0,
        boundary_pos_weight: float = 5.0,
        skeleton_iterations: int = 5,
    ):
        super().__init__()
        if min(bce_weight, dice_weight, boundary_weight, cldice_weight) < 0:
            raise ValueError("Loss weights cannot be negative")
        if bce_weight + dice_weight + boundary_weight + cldice_weight == 0:
            raise ValueError("At least one loss weight must be positive")
        if skeleton_iterations < 0:
            raise ValueError("skeleton_iterations cannot be negative")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.cldice_weight = cldice_weight
        self.skeleton_iterations = skeleton_iterations
        if boundary_pos_weight <= 0:
            raise ValueError("boundary_pos_weight must be positive")
        self.register_buffer(
            "boundary_pos_weight", torch.tensor(float(boundary_pos_weight))
        )

    def forward(self, logits: torch.Tensor, targets):
        """Accept explicit foreground/separation/valid BCHW maps, or legacy IDs.

        The explicit maps preserve overlap foreground; integer-ID compatibility
        cannot represent overlapping masks and is for old callers only.
        """
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise ValueError("Expected logits [B, 2, H, W]")
        if isinstance(targets, torch.Tensor):
            if targets.ndim != 3:
                raise ValueError("Expected legacy instance IDs BHW")
            foreground = (targets != 0).unsqueeze(1).float()
            boundaries = instance_boundaries(targets)
            valid = (targets >= 0).unsqueeze(1).float()
        else:
            foreground, boundaries, valid = (targets[k].float() for k in
                ("foreground", "separation", "separation_valid"))
        if any(t.shape != logits[:, :1].shape for t in (foreground, boundaries, valid)):
            raise ValueError("All loss target maps must be B1HW aligned with logits")
        # Explicitly disable ambient autocast: sums and BCE remain FP32.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            mask_logits, boundary_logits = logits[:, :1].float(), logits[:, 1:2].float()
            probabilities = torch.sigmoid(mask_logits)
            bce = F.binary_cross_entropy_with_logits(mask_logits, foreground)
            axes = (1, 2, 3)
            intersection = (probabilities * foreground).sum(dim=axes)
            denominator = probabilities.sum(dim=axes) + foreground.sum(dim=axes)
            dice = (1 - (2 * intersection + 1.0) / (denominator + 1.0)).mean()
            boundary_pixels = F.binary_cross_entropy_with_logits(
                boundary_logits, boundaries, pos_weight=self.boundary_pos_weight, reduction="none")
            # Per-image reduction makes uneven accumulation windows equivalent
            # to the mean of their samples even when overlap exclusions vary.
            boundary_bce = ((boundary_pixels * valid).sum(axes) / valid.sum(axes).clamp_min(1)).mean()
            cldice = soft_cldice_loss(probabilities, foreground, iterations=self.skeleton_iterations
            ) if self.cldice_weight > 0 else mask_logits.new_zeros(())
            total = (
                self.bce_weight * bce + self.dice_weight * dice
                + self.boundary_weight * boundary_bce + self.cldice_weight * cldice
            )
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice": dice.detach(),
            "boundary": boundary_bce.detach(),
            "cldice": cldice.detach(),
        }


if __name__ == "__main__":
    # Small smoke test, including a 2-instance target and backpropagation.
    model_output = torch.randn(2, 2, 63, 65, requires_grad=True)
    instances = torch.zeros(2, 63, 65, dtype=torch.long)
    instances[0, 5:35, 8:12] = 1
    instances[0, 35:58, 8:12] = 2  # Touching filament instances
    instances[1, 12:48, 20:25] = 1

    criterion = FilamentPQProxyLoss(cldice_weight=0.10)
    values = criterion(model_output, instances)
    values["loss"].backward()

    assert torch.isfinite(values["loss"])
    assert model_output.grad is not None
    assert torch.isfinite(model_output.grad).all()
    print({k: round(v.item(), 4) for k, v in values.items()})
    print("Backward pass: OK")
