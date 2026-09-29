"""
Consistency Regularization Loss and Ramp-up Schedulers for Semi-Supervised Learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_rampup(current: float, rampup_length: float) -> float:
    """Exponential ramp-up schedule used in Mean Teacher and consistency regularizers."""
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


class ConsistencyLoss(nn.Module):
    """
    Consistency Loss penalizing divergence between student and teacher predictions on unlabeled data.
    Supports Mean Squared Error (MSE) and symmetric Kullback-Leibler (KL) divergence.
    """
    def __init__(self, mode: str = 'mse'):
        super().__init__()
        self.mode = mode

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)

        if self.mode == 'mse':
            return F.mse_loss(student_probs, teacher_probs)
        elif self.mode == 'kl_div':
            log_student = F.log_softmax(student_logits, dim=1)
            return F.kl_div(log_student, teacher_probs, reduction='batchmean')
        else:
            raise ValueError(f"Unsupported consistency mode: {self.mode}")
