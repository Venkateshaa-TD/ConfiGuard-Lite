"""Ground-truth BCE plus optional binary logit distillation.

loss = (1 - alpha) * BCE(z, y) + alpha * T^2 * BCE(sigmoid(z / T), sigmoid(m / T))

z is the student's single logit and m the cached teacher margin
(logit_fake - logit_real), so sigmoid(m) is exactly GenD's softmax P(fake).
The T^2 factor keeps the soft-target gradient scale comparable across
temperatures (Hinton et al., 2015). alpha = 0 is the plain baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DistillLossParts:
    total: torch.Tensor
    hard: torch.Tensor
    soft: torch.Tensor


def distillation_loss(student_logit: torch.Tensor, labels: torch.Tensor, teacher_margin: torch.Tensor,
                      alpha: float, temperature: float) -> DistillLossParts:
    if not 0.0 <= alpha <= 1.0 or temperature <= 0:
        raise ValueError(f"alpha must be in [0, 1] and temperature > 0 (got {alpha}, {temperature})")
    z = student_logit.float()
    hard = F.binary_cross_entropy_with_logits(z, labels.float())
    if alpha == 0.0:
        return DistillLossParts(hard, hard, torch.zeros_like(hard))
    soft_target = torch.sigmoid(teacher_margin.float() / temperature)
    soft = F.binary_cross_entropy_with_logits(z / temperature, soft_target) * temperature**2
    return DistillLossParts((1.0 - alpha) * hard + alpha * soft, hard, soft)
