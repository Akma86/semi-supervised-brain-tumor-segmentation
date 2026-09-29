"""
Mean Teacher semi-supervised framework for brain tumor segmentation.
Maintains a Student model and an EMA (Exponential Moving Average) Teacher model.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from .consistency import ConsistencyLoss, sigmoid_rampup
from ..models.segmentation_heads import DiceCELoss


class MeanTeacherFramework:
    """
    Mean Teacher Architecture:
    - Student model learns from labeled supervised loss + unlabeled consistency loss.
    - Teacher model weights are updated as the Exponential Moving Average of the student weights.
    """
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema_decay: float = 0.99,
        consistency_weight: float = 1.0,
        rampup_epochs: int = 20,
        device: str = "cpu"
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.optimizer = optimizer
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight
        self.rampup_epochs = rampup_epochs
        self.device = device

        # Initialize teacher weights to match student
        self.update_ema_weights(decay=0.0)

        # Detach teacher parameters from autograd
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.supervised_criterion = DiceCELoss()
        self.consistency_criterion = ConsistencyLoss(mode='mse')

    def update_ema_weights(self, decay: float):
        """Updates teacher model weights using EMA formula: theta_t = decay * theta_t + (1 - decay) * theta_s"""
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

    def train_step(
        self,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Dict[str, torch.Tensor],
        current_epoch: int
    ) -> Dict[str, float]:
        """
        Executes a single semi-supervised optimization step.
        """
        self.student.train()
        self.teacher.train()

        img_l = labeled_batch['image'].to(self.device)
        target_l = labeled_batch['mask'].to(self.device)
        img_u = unlabeled_batch['image'].to(self.device)

        # 1. Supervised Forward Pass on Labeled Data
        pred_l = self.student(img_l)
        loss_sup = self.supervised_criterion(pred_l, target_l)

        # 2. Unsupervised Forward Pass on Unlabeled Data
        # Add slight perturbation/noise to student input to enforce consistency
        noise = torch.randn_like(img_u) * 0.05
        student_unlabeled_pred = self.student(img_u + noise)

        with torch.no_grad():
            teacher_unlabeled_pred = self.teacher(img_u)

        # 3. Consistency Regularization Loss
        loss_const = self.consistency_criterion(student_unlabeled_pred, teacher_unlabeled_pred)

        # 4. Total Loss with Dynamic Ramp-up Weight
        current_rampup = sigmoid_rampup(current_epoch, self.rampup_epochs)
        current_consistency_weight = self.consistency_weight * current_rampup
        total_loss = loss_sup + current_consistency_weight * loss_const

        # 5. Backpropagation & Optimizer Step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 6. Update Teacher EMA Weights
        self.update_ema_weights(decay=self.ema_decay)

        return {
            'total_loss': total_loss.item(),
            'loss_supervised': loss_sup.item(),
            'loss_consistency': loss_const.item(),
            'consistency_weight': current_consistency_weight
        }
