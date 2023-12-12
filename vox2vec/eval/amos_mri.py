import os
import json
import nibabel
import numpy as np
from typing import *
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from connectome import Source, meta, Chain, Transform, Apply, CacheToRam, CacheToDisk

from vox2vec.utils.box import mask_to_bbox
from vox2vec.utils.data import VanillaDataset, Pool
from vox2vec.processing import (
    scale_0_1,
    get_body_mask,
    sample_patches,
    CropToBox,
    LabelsToOnehot
)


LABELS = {
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

# DATA = [
#     'annotations/amos_mri_fold0.json',
#     'annotations/amos_mri_fold1.json',
#     'annotations/amos_mri_fold2.json',
#     'annotations/amos_mri_fold3.json',
#     'annotations/amos_mri_fold4.json'
# ]

DATA = [
    'annotations/amos_mri_4_fold0.json',
    'annotations/amos_mri_4_fold1.json',
    'annotations/amos_mri_4_fold2.json',
    'annotations/amos_mri_4_fold3.json',
    'annotations/amos_mri_4_fold4.json'
]

# DATA = [
#     'annotations/amos_mri_8_fold0.json',
#     'annotations/amos_mri_8_fold1.json',
#     'annotations/amos_mri_8_fold2.json',
#     'annotations/amos_mri_8_fold3.json',
#     'annotations/amos_mri_8_fold4.json'
# ]

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

class AMOSMRISource(Source):
    _root: str
    _split: int

    @meta
    def train_ids(_root, _split):
        train_files = read_json_data_file(os.path.join(_root, 'dataset_preprocessed_mri', DATA[_split]), _root, 'training')
        return sorted({
            Path(file['image']).name[len('amos_'):-len('.nii.gz')]
            for file in train_files
        })

    @meta
    def val_ids(_root, _split):
        val_files = read_json_data_file(os.path.join(_root, 'dataset_preprocessed_mri', DATA[_split]), _root, 'validation')
        return sorted({
            Path(file['image']).name[len('amos_'):-len('.nii.gz')]
            for file in val_files
        })

    @meta
    def test_ids(_root,  _split):
        test_files = read_json_data_file(os.path.join(_root, 'dataset_preprocessed_mri', DATA[_split]), _root, 'test')
        return sorted({
            Path(file['image']).name[len('amos_'):-len('.nii.gz')]
            for file in test_files
        })

    def _image_nii(id_, _root):
        file,  = Path(_root).glob(f'dataset_preprocessed_mri/images[TV][ra]/amos_{id_}.nii.gz')
        return nibabel.load(file)

    def _mask_nii(id_, _root):
        try:
            file, = Path(_root).glob(f'dataset_preprocessed_mri/labels[TV][ra]/amos_{id_}.nii.gz')
        except ValueError:
            return

        return nibabel.load(file)

    def image(_image_nii):
        img = _image_nii.get_fdata().astype(np.float32)
        return img

    def affine(_image_nii):
        return _image_nii.affine

    def mask(_mask_nii):
        if _mask_nii is not None:
            return _mask_nii.get_fdata().astype(np.int16)


class AMOSMRI(pl.LightningDataModule):
    num_classes = len(LABELS)

    def __init__(
            self,
            root: str,
            cache_dir: str,
            patch_size: Tuple[int, int, int],
            batch_size: int,
            num_batches_per_epoch: Optional[int],
            num_workers: int,
            buffer_size: int,
            split: int,
            cache_to_ram: bool = True
    ) -> None:
        super().__init__()

        source = AMOSMRISource(root=root, split=split)

        train_ids = source.train_ids
        val_ids = source.val_ids
        test_ids = source.test_ids

        # use connectome for smart cashing (with automatic invalidation)
        preprocessing = Chain(
            Apply(image=scale_0_1),
            Transform(__inherit__=True, cropping_box=lambda image: mask_to_bbox(image > 0)),
            CropToBox(axis=(-3, -2, -1)),
            Transform(__inherit__=True, body_mask=lambda image: get_body_mask(image, 0))
        )

        train_pipeline = source >> preprocessing >> CacheToDisk.simple('image', 'body_mask', 'mask', root=cache_dir)
        if cache_to_ram:
            train_pipeline >>= CacheToRam(['image', 'body_mask', 'mask'])
        train_pipeline >>= LabelsToOnehot(labels_range=range(1, len(LABELS) + 1))
        train_pipeline >>= Apply(image=lambda x: x[None], mask=np.float32)

        _load_train_example = train_pipeline._compile(['image', 'body_mask', 'mask'])

        def load_train_example(id_):
            examples = sample_patches(*_load_train_example(id_), patch_size, batch_size)
            return [torch.tensor(np.stack(xs)) for xs in zip(*examples)]

        load_val_example = load_test_example = _load_train_example

        self.train_dataset = Pool(
            VanillaDataset(train_ids, load_train_example),
            num_samples=num_batches_per_epoch,
            num_workers=num_workers,
            buffer_size=buffer_size
        )
        self.val_dataset = VanillaDataset(val_ids, load_val_example)
        self.test_dataset = VanillaDataset(test_ids, load_test_example)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=None)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=None)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=None)
