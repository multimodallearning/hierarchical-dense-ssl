from typing import *

import torch
from torch import nn

import pytorch_lightning as pl

from .predict import predict
from vox2vec.nn.functional import (
    eval_mode,
    compute_dice_score,
    compute_binary_segmentation_loss
)



class Probing(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            *heads: nn.Module,
            patch_size: Tuple[int, int, int],
            threshold: float = 0.5,
            lr: float = 3e-4,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['backbone', 'heads'])

        self.backbone = backbone
        self.heads = nn.ModuleList(heads)

        self.patch_size = patch_size
        self.threshold = threshold
        self.lr = lr

        self.automatic_optimization = False

        self.val_dice = [0, 0, 0, 0, 0, 0]

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        images, rois, gt_masks = batch

        # image_original_np = rois.data.cpu().numpy()[0, ...].copy()
        # center_slice = image_original_np.shape[2] // 2
        # for image_slice in range(center_slice, center_slice + 1):
        #     plt.imshow(image_original_np[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()

        with torch.no_grad(), eval_mode(self.backbone):
            backbone_outputs = self.backbone(images)

        for i, head in enumerate(self.heads):
            pred_logits = head(backbone_outputs)
            # loss, logs = compute_binary_segmentation_loss(pred_logits, gt_masks, rois, logs_prefix=f'train/head_{i}_')
            loss, logs = compute_binary_segmentation_loss(pred_logits, gt_masks, None, logs_prefix=f'train/head_{i}_')  # Uncomment for MRI data
            self.log_dict(logs, on_epoch=True, on_step=False)
            self.manual_backward(loss)

        optimizer.step()

    def validation_step(self, batch, batch_idx):
        image, roi, gt_mask = batch

        for i, head in enumerate(self.heads):
            # pred_probas = predict(image, self.patch_size, self.backbone, head, self.device, roi)
            pred_probas = predict(image, self.patch_size, self.backbone, head, self.device, None)  # Uncomment for MRI data
            dice_scores = compute_dice_score(pred_probas, gt_mask, reduce=lambda x: x)
            for j, dice_score in enumerate(dice_scores):
                self.log(f'val/head_{i}_dice_score_for_cls_{j}', dice_score, on_epoch=True)
            self.log(f'val/head_{i}_avg_dice_score', dice_scores.mean(), on_epoch=True)

            if i == 0:
                if self.val_dice[batch_idx] != 0:
                    self.val_dice[batch_idx] = 0.7 * self.val_dice[batch_idx] + 0.3 * dice_scores.mean()
                else:
                    self.val_dice[batch_idx] = dice_scores.mean()
                self.log(f'val/head_0_smooth_dice_score', self.val_dice[batch_idx], on_epoch=True)


    def test_step(self, batch, batch_idx):
        image, roi, gt_mask = batch
        for i, head in enumerate(self.heads):
            # pred_probas = predict(image, self.patch_size, self.backbone, head, self.device, roi)
            pred_probas = predict(image, self.patch_size, self.backbone, head, self.device, None)  # Uncomment for MRI data
            pred_mask = pred_probas >= self.threshold
            dice_scores = compute_dice_score(pred_mask, gt_mask, reduce=lambda x: x)
            for j, dice_score in enumerate(dice_scores):
                self.log(f'test/head_{i}_dice_score_for_cls_{j}', dice_score, on_epoch=True)
            self.log(f'test/head_{i}_avg_dice_score', dice_scores.mean(), on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val and test dataloaders
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
