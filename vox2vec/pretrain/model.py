from typing import *

import logging
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from vox2vec.nn import Lambda
from vox2vec.nn.functional import select_from_pyramid


class Vox2Vec(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            base_channels: int,
            num_scales: int,
            proj_dim: int = 128,
            temp: float = 0.1,
            lr: float = 3e-4,
    ):
        """vox2vec model.

        Args:
            backbone (nn.Module):
                Takes an image of size ``(n, c, h, w, d)`` and returns a feature pyramid of sizes
                ``[(n, c_b, h_b, w_b, d_b), (n, c_b * 2, h_b // 2, w_b // 2, d_b // 2), ...]``,
                where ``c_b = base_channels`` and ``(h_b, w_b, d_b) = (h, w, d)``.
            base_channels (int):
                A number of channels in the base of the output feature pyramid.
            num_scales (int):
                A number of feature pyramid levels.
            proj_dim (int, optional):
                The output dimensionality of the projection head. Defaults to 128.
            temp (float, optional):
                Info-NCE loss temperature. Defaults to 0.1.
            lr (float, optional):
                Learning rate. Defaults to 3e-4.
        """
        super().__init__()

        self.save_hyperparameters(ignore='backbone')

        self.backbone = backbone
        embed_dim = 160
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
            Lambda(F.normalize)
        )

        # self.proj_heads = [
        #     nn.Sequential(
        #         nn.Linear(base_channels * 2 ** i, base_channels * 2 ** i).cuda(),
        #         nn.BatchNorm1d(base_channels * 2 ** i).cuda(),
        #         nn.ReLU().cuda(),
        #         nn.Linear(base_channels * 2 ** i, base_channels * 2 ** i).cuda(),
        #         nn.BatchNorm1d(base_channels * 2 ** i).cuda(),
        #         nn.ReLU().cuda(),
        #         nn.Linear(base_channels * 2 ** i, min(base_channels * 2 ** i, 128)).cuda(),
        #         Lambda(F.normalize).cuda()
        #     )
        #     for i in range(num_scales)
        # ]

        self.scale_proj = [nn.Conv3d(base_channels * 2 ** i, 32, kernel_size=1, bias=(i == 0)).cuda() for i in range(num_scales)]

        # self.scale_proj = [
        #     nn.Sequential(
        #         nn.Upsample(size=(128, 128, 32), mode='trilinear').cuda()
        #     )
        #     for i in range(num_scales)
        # ]

        self.rest_head = nn.Conv3d(base_channels, 1, kernel_size=1)

        # self.rest_head = nn.Sequential(
        #     nn.BatchNorm3d(base_channels),
        #     nn.ReLU(),
        #     nn.Conv3d(base_channels, base_channels // 2, 3, padding=1),
        #     nn.BatchNorm3d(base_channels // 2),
        #     nn.ReLU(),
        #     nn.Conv3d(base_channels // 2, 1, 1)
        #     )

        self.temp = temp
        self.lr = lr

    def _vox_to_vec(self, feature_pyramid: Iterable[torch.Tensor], voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])

    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1_aug, patches_2_aug, patches_1, patches_2, voxels_1, voxels_2 = batch['pretrain']

        # image_original_np = patches_1.data.cpu().numpy()[0, 0, ...].copy()
        # center_slice = image_original_np.shape[2] // 2
        # for image_slice in range(center_slice, center_slice + 1):
        #     plt.imshow(image_original_np[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()
        #
        # image_original_np = patches_1_aug.data.cpu().numpy()[0, 0, ...].copy()
        # center_slice = image_original_np.shape[2] // 2
        # for image_slice in range(center_slice, center_slice + 1):
        #     plt.imshow(image_original_np[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()
        #
        # image_original_np = patches_2.data.cpu().numpy()[0, 0, ...].copy()
        # center_slice = image_original_np.shape[2] // 2
        # for image_slice in range(center_slice, center_slice + 1):
        #     plt.imshow(image_original_np[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()
        #
        # image_original_np = patches_2_aug.data.cpu().numpy()[0, 0, ...].copy()
        # center_slice = image_original_np.shape[2] // 2
        # for image_slice in range(center_slice, center_slice + 1):
        #     plt.imshow(image_original_np[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()

        assert self.backbone.training
        assert self.proj_head.training
        assert self.rest_head.training

        feature_pyramid_1 = self.backbone(patches_1_aug)
        feature_pyramid_1_proj = [self.scale_proj[i](feature_pyramid_1[i]) for i in range(len(feature_pyramid_1))]
        # embeds_1 = [self.proj_heads[i](self._vox_to_vec([feature_pyramid_1_proj[i]], voxels_1)) for i in range(len(feature_pyramid_1))]
        embeds_1 = self.proj_head(self._vox_to_vec(feature_pyramid_1_proj, voxels_1))
        rest_patches_1 = self.rest_head(feature_pyramid_1[0])

        feature_pyramid_2 = self.backbone(patches_2_aug)
        feature_pyramid_2_proj = [self.scale_proj[i](feature_pyramid_2[i]) for i in range(len(feature_pyramid_2))]
        # embeds_2 = [self.proj_heads[i](self._vox_to_vec([feature_pyramid_2_proj[i]], voxels_2)) for i in range(len(feature_pyramid_2))]
        embeds_2 = self.proj_head(self._vox_to_vec(feature_pyramid_2_proj, voxels_2))
        rest_patches_2 = self.rest_head(feature_pyramid_2[0])

        # InfoNCE loss

        logits_11 = torch.matmul(embeds_1, embeds_1.T) / self.temp
        logits_11.fill_diagonal_(float('-inf'))
        logits_12 = torch.matmul(embeds_1, embeds_2.T) / self.temp
        logits_22 = torch.matmul(embeds_2, embeds_2.T) / self.temp
        logits_22.fill_diagonal_(float('-inf'))
        info_nce_loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        info_nce_loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
        info_nce_loss_total = (info_nce_loss_1 + info_nce_loss_2) / 2

        # info_nce_loss_total = 0
        # for i in range(len(feature_pyramid_1)):
        #     logits_11 = torch.matmul(embeds_1[i], embeds_1[i].T) / self.temp
        #     logits_11.fill_diagonal_(float('-inf'))
        #     logits_12 = torch.matmul(embeds_1[i], embeds_2[i].T) / self.temp
        #     logits_22 = torch.matmul(embeds_2[i], embeds_2[i].T) / self.temp
        #     logits_22.fill_diagonal_(float('-inf'))
        #     info_nce_loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        #     info_nce_loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
        #     info_nce_loss = (info_nce_loss_1 + info_nce_loss_2) / 2
        #
        #     self.log(f'pretrain/info_nce_loss_scale{i}', info_nce_loss, on_epoch=True)
        #
        #     info_nce_loss_total += info_nce_loss

        # MSE loss
        mse_loss = (torch.mean((rest_patches_1 - patches_1) ** 2) + torch.mean((rest_patches_2 - patches_2) ** 2)) / 2

        loss = info_nce_loss_total + 10 * mse_loss

        self.log(f'pretrain/info_nce_loss', info_nce_loss_total, on_epoch=True)
        self.log(f'pretrain/mse_loss', mse_loss, on_epoch=True)
        self.log(f'pretrain/total_loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):

        if not self.trainer.training: # skip device transfer for the val dataloader
            return batch

        return super().transfer_batch_to_device(batch, device, dataloader_idx)
