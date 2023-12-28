import os
import json
import nibabel
import numpy as np
from typing import *
from pathlib import Path
from imops import crop_to_box
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset

from vox2vec.utils.box import mask_to_bbox
from vox2vec.processing import BODY_THRESHOLD, sample_patches

AMOS_DATA_DIR = '/home/kats/storage/staff/eytankats/data/hierarchical_dense_ssl/downstream/amos/'
AMOS_LABELS = {
    1: 'Spleen',
    2: 'Right kidney',
    3: 'Left kidney',
    4: 'Gallbladder',
    5: 'Esophagus',
    6: 'Liver',
    7: 'Stomach',
    8: 'Aorta',
    9: 'Inferior Vena Cava (IVC)',
    10: 'Pancreas',
    11: 'Right Adrenal Gland',
    12: 'Left Adrenal Gland',
}
AMOS_DATA_SPLIT = [
    'splits/amos_mri_fold0.json',
    'splits/amos_mri_fold1.json',
    'splits/amos_mri_fold2.json',
    'splits/amos_mri_fold3.json',
    'splits/amos_mri_fold4.json'
]
AMOS_DATA_SPLIT_4 = [
    'splits/amos_mri_4_fold0.json',
    'splits/amos_mri_4_fold1.json',
    'splits/amos_mri_4_fold2.json',
    'splits/amos_mri_4_fold3.json',
    'splits/amos_mri_4_fold4.json'
]
AMOS_DATA_SPLIT_8 = [
    'splits/amos_mri_8_fold0.json',
    'splits/amos_mri_8_fold1.json',
    'splits/amos_mri_8_fold2.json',
    'splits/amos_mri_8_fold3.json',
    'splits/amos_mri_8_fold4.json'
]

BTCV_DATA_DIR = '/home/kats/storage/staff/eytankats/data/hierarchical_dense_ssl/downstream/btcv/'
BTCV_LABELS = {
    1: 'Spleen',
    2: 'Right kidney',
    3: 'Left kidney',
    4: 'Gallbladder',
    5: 'Esophagus',
    6: 'Liver',
    7: 'Stomach',
    8: 'Aorta',
    9: 'Inferior Vena Cava (IVC)',
    10: 'Portal Vein and Splenic Vein',
    11: 'Pancreas',
    12: 'Right Adrenal Gland',
    13: 'Left Adrenal Gland',
}
BTCV_DATA_SPLIT = [
    'splits/btcv_fold0_tr20.json',
    'splits/btcv_fold1_tr20.json',
    'splits/btcv_fold2_tr20.json'
]

def read_json_data_file(
        data_file_path,
        data_dir,
        key='training'
):

    with open(data_file_path) as f:
        json_data = json.load(f)

    tr = []
    json_data_training = json_data[key]
    for d in json_data_training:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(data_dir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(data_dir, d[k]) if len(d[k]) > 0 else d[k]
        tr.append(d)

    return tr


def get_data_ids(
        data_file_path,
        key='training'
):
    files = read_json_data_file(data_file_path, '', key)
    ids = sorted({Path(file['image']).name for file in files})
    return ids

def pad_to_min_shape(a, min_shape):
    x_, y_, z_, = min_shape
    x, y, z = a.shape
    x_pad = max(0, (x_-x))
    y_pad = max(0, (y_-y))
    z_pad = max(0, (z_-z))
    return np.pad(a,((x_pad//2, x_pad//2 + x_pad%2), (y_pad//2, y_pad//2 + y_pad%2), (z_pad//2, z_pad//2 + z_pad%2)), mode='constant')

def labels_to_onehot(labels: np.ndarray, labels_range: Iterable[int]):
    return np.stack([labels == lbl_idx for lbl_idx in labels_range]).astype(np.float32)


class DownstreamDataset(Dataset):

    def __init__(
            self,
            dataset: str,
            patch_size: Tuple[int, int, int],
            batch_size: int,
            batches_per_epoch: int,
            split: int,
            mode: str,
            examples_num: str = 'all',
    ) -> None:

        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.batches_per_epoch = batches_per_epoch
        self.mode = mode

        if dataset == 'btcv':
            self.data_dir = BTCV_DATA_DIR
            data_file_path = os.path.join(BTCV_DATA_DIR, BTCV_DATA_SPLIT[split])
        elif dataset == 'amos' and examples_num == 'all':
            self.data_dir = AMOS_DATA_DIR
            data_file_path = os.path.join(AMOS_DATA_DIR, AMOS_DATA_SPLIT[split])
        elif dataset == 'amos' and examples_num == '8':
            self.data_dir = AMOS_DATA_DIR
            data_file_path = os.path.join(AMOS_DATA_DIR, AMOS_DATA_SPLIT_8[split])
        elif dataset == 'amos' and examples_num == '4':
            self.data_dir = AMOS_DATA_DIR
            data_file_path = os.path.join(AMOS_DATA_DIR, AMOS_DATA_SPLIT_4[split])

        self.ids = get_data_ids(data_file_path, mode)

        if dataset == 'btcv':
            self.num_classes = len(BTCV_LABELS)
        elif dataset == 'amos':
            self.num_classes = len(AMOS_LABELS)

    def load_example(self, image_path, mask_path):
        image = nibabel.load(image_path).get_fdata().astype(np.float32)
        mask = nibabel.load(mask_path).get_fdata().astype(np.float32)

        box = mask_to_bbox(image >= BODY_THRESHOLD, self.patch_size)
        image = crop_to_box(image, box, axis=(-3, -2, -1))
        mask = crop_to_box(mask, box, axis=(-3, -2, -1))

        image = pad_to_min_shape(image, self.patch_size)
        mask = pad_to_min_shape(mask, self.patch_size)

        center_slice = image.shape[2] // 2

        plt.imshow(image[:, :, center_slice], cmap='gray')
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(mask[:, :, center_slice])
        plt.colorbar()
        plt.show()
        plt.close()

        mask = labels_to_onehot(mask, labels_range=range(1, self.num_classes + 1))

        return image, mask

    def __len__(self):
        if self.mode == 'training':
            length = self.batches_per_epoch
        else:
            length = len(self.ids)

        return length

    def __getitem__(self, i):

        if self.mode == 'training':
            idx = np.random.randint(0, len(self.ids))
        else:
            idx = i

        image_path = os.path.join(self.data_dir, 'images', self.ids[idx])

        if self.dataset == 'amos':
            mask_path = os.path.join(self.data_dir, 'labels', self.ids[idx])
        elif self.dataset == 'btcv':
            mask_path = os.path.join(self.data_dir, 'labels', 'label' + self.ids[idx][3:])

        image, mask = self.load_example(image_path, mask_path)

        if self.mode == 'training':
            patches_iter = sample_patches(image, np.zeros_like(image), mask, self.patch_size, self.batch_size)
            return [torch.tensor(np.stack(xs)) for xs in zip(*patches_iter)]

        return image, np.zeros_like(image), mask






