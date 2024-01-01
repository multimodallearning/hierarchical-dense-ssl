import random
import nibabel
import numpy as np
from typing import *
from pathlib import Path
from imops import crop_to_box
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset

from vox2vec.utils.box import mask_to_bbox
from vox2vec.utils.intensity_augmentations import AppearanceTransform
from vox2vec.processing import BODY_THRESHOLD, sample_box, get_body_mask

NAKO_DATA_DIR = '/path/to/preprocessed_nako_dataset/'
AMOS_DATA_DIR = '/path/to/preprocessed_amos_ct_dataset/'
FLARE_DATA_DIR = '/path/to/preprocessed_flare_dataset/'

class PretrainingDataset(Dataset):

    def __init__(
            self,
            patch_size: Tuple[int, int, int],
            max_num_voxels_per_patch: int,
            batch_size: int,
            batches_per_epoch: int,
            pretraining_dataset: str,
    ) -> None:

        if pretraining_dataset == 'nako':
            self.data_paths = [data_path for data_path in Path(NAKO_DATA_DIR).glob('*.nii.gz')]
        elif pretraining_dataset == 'flare_amos':
            self.data_paths = ([data_path for data_path in Path(AMOS_DATA_DIR).glob('*.nii.gz')] +
                               [data_path for data_path in Path(FLARE_DATA_DIR).glob('*.nii.gz')])

        self.patch_size = patch_size
        self.max_num_voxels_per_patch = max_num_voxels_per_patch
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        self.style_aug = AppearanceTransform(
            local_rate=0.8,
            nonlinear_rate=0.9,
            paint_rate=0.9,
            inpaint_rate=0.2
        )

    def load_example(self, data_path):
        image = nibabel.load(data_path).get_fdata().astype(np.float32)

        box = mask_to_bbox(image >= BODY_THRESHOLD, self.patch_size)
        image = crop_to_box(image, box, axis=(-3, -2, -1))

        # center_slice = image.shape[2] // 2
        # plt.imshow(image[:, :, center_slice], cmap='gray')
        # plt.colorbar()
        # plt.show()
        # plt.close()

        voxels = np.argwhere(get_body_mask(image, BODY_THRESHOLD))
        return image, voxels

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, i):

        idx = np.random.randint(0, len(self.data_paths))

        args = [*self.load_example(self.data_paths[idx]), self.patch_size, self.max_num_voxels_per_patch, self.style_aug]
        views = [sample_views(*args) for _ in range(self.batch_size)]
        patches_1_aug, patches_2_aug, patches_1, patches_2, voxels_1, voxels_2 = zip(*views)
        patches_1 = torch.tensor(np.stack([p[None] for p in patches_1]))
        patches_2 = torch.tensor(np.stack([p[None] for p in patches_2]))
        patches_1_aug = torch.tensor(np.stack([p[None] for p in patches_1_aug]))
        patches_2_aug = torch.tensor(np.stack([p[None] for p in patches_2_aug]))
        voxels_1 = [torch.tensor(voxels) for voxels in voxels_1]
        voxels_2 = [torch.tensor(voxels) for voxels in voxels_2]
        return patches_1_aug, patches_2_aug, patches_1, patches_2, voxels_1, voxels_2


def sample_views(
        image: np.ndarray,
        roi_voxels: np.ndarray,
        patch_size: Tuple[int, int, int],
        max_num_voxels: int,
        style_aug
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_voxel = random.choice(roi_voxels)

    patch_1_aug, patch_1, roi_voxels_1 = sample_view(image, roi_voxels, anchor_voxel, patch_size, style_aug)
    patch_2_aug, patch_2, roi_voxels_2 = sample_view(image, roi_voxels, anchor_voxel, patch_size, style_aug)

    valid_1 = np.all((roi_voxels_1 >= 0) & (roi_voxels_1 < patch_size), axis=1)
    valid_2 = np.all((roi_voxels_2 >= 0) & (roi_voxels_2 < patch_size), axis=1)
    valid = valid_1 & valid_2
    assert valid.any()
    indices = np.where(valid)[0]

    if len(indices) > max_num_voxels:
        indices = np.random.choice(indices, max_num_voxels, replace=False)

    return patch_1_aug, patch_2_aug, patch_1, patch_2, roi_voxels_1[indices], roi_voxels_2[indices]


def sample_view(image, voxels, anchor_voxel, patch_size, style_aug):
    assert image.ndim == 3

    box = sample_box(image.shape, patch_size, anchor_voxel)
    image = crop_to_box(image, box, axis=(-3, -2, -1))
    shift = box[0]
    voxels = voxels - shift

    image_aug = style_aug.rand_aug(image.copy())

    return image_aug, image, voxels
