from typing import *
import numpy as np

import torch
from torch import nn

import pytorch_lightning as pl

from vox2vec.nn.functional import (
    compute_binary_segmentation_loss, compute_dice_score
)
from .predict import predict


class EndToEnd(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            head: nn.Module,
            patch_size: Tuple[int, int, int],
            threshold: float = 0.5,
            lr: float = 3e-4,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['backbone', 'head'])

        self.backbone = backbone
        self.head = head
        
        self.patch_size = patch_size
        self.threshold = threshold
        self.lr = lr

        self.val_dice = [0, 0, 0, 0, 0, 0]

        self.test_results = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def predict(self, image: torch.Tensor, roi: Optional[torch.Tensor] = None) -> torch.Tensor:
        return predict(image, self.patch_size, self.backbone, self.head, self.device, roi)

    def training_step(self, batch, batch_idx):
        images, rois, gt_masks = batch
        loss, logs = compute_binary_segmentation_loss(self(images), gt_masks, rois, logs_prefix='train/')
        self.log_dict(logs, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        image, roi, gt_mask = batch
        pred_probas = self.predict(image, roi)
        dice_scores = compute_dice_score(pred_probas, gt_mask, reduce=lambda x: x)
        for i, dice_score in enumerate(dice_scores):
            self.log(f'val/head_0_dice_score_for_cls_{i}', dice_score, on_epoch=True)
        self.log(f'val/head_0_avg_dice_score', dice_scores.mean(), on_epoch=True)

        if self.val_dice[batch_idx] != 0:
            self.val_dice[batch_idx] = 0.7 * self.val_dice[batch_idx] + 0.3 * dice_scores.mean()
        else:
            self.val_dice[batch_idx] = dice_scores.mean()
        self.log(f'val/head_0_smooth_dice_score', self.val_dice[batch_idx], on_epoch=True)

    def test_step(self, batch, batch_idx):
        image, roi, gt_mask = batch
        pred_probas = self.predict(image, roi)
        pred_mask = pred_probas >= self.threshold
        dice_scores = compute_dice_score(pred_mask, gt_mask, reduce=lambda x: x)

        if self.test_results is None:
            self.test_results = dice_scores
        else:
            self.test_results = np.vstack([self.test_results, dice_scores])

        for i, dice_score in enumerate(dice_scores):
            self.log(f'test/head_0_dice_score_for_cls_{i}', dice_score)
        self.log(f'test/head_0_avg_dice_score', dice_scores.mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val and test dataloaders
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
