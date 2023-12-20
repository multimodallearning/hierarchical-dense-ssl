# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from argparse import ArgumentParser

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from torch.utils.data import DataLoader

from clearml import Task

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from vox2vec.default_params import *
from vox2vec.pretrain.pretraining_dataset import PretrainingDataset
from vox2vec.eval.btcv import BTCV
from vox2vec.eval.amos_mri_conventional import AMOSMRI
from vox2vec.nn import FPN3d, FPNLinearHead, FPNNonLinearHead
from vox2vec.pretrain.model import Vox2Vec
from vox2vec.eval.online_probing import OnlineProbing


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--probing_dataset', default='amos_mri')
    parser.add_argument('--pretraining_dataset', default='nako')

    parser.add_argument('--cache_dir', default='/home/kats/share/experiments/label/vox2vec/debug/data_cache')
    parser.add_argument('--log_dir', default='/home/kats/share/experiments/label/vox2vec/debug')
    parser.add_argument('--root_data_dir', default='/home/kats/storage/staff/eytankats/data/hierarchical_dense_ssl/pretraining/nako/images')
    parser.add_argument('--amos_dir', default='/home/kats/storage/staff/eytankats/data/hierarchical_dense_ssl/pretraining/amos/images')
    parser.add_argument('--flare_dir', default='/home/kats/storage/staff/eytankats/data/hierarchical_dense_ssl/pretraining/flare/images')
    # parser.add_argument('--probing_data_dir', default='/mnt/share/data/amos')

    parser.add_argument('--spacing', nargs='+', type=float, default=SPACING)
    parser.add_argument('--patch_size', nargs='+', type=int, default=PATCH_SIZE)
    parser.add_argument('--pretrain_batch_size', type=int, default=10)
    parser.add_argument('--pretrain_num_workers', type=int, default=4)
    parser.add_argument('--probing_batch_size', type=int, default=5)
    parser.add_argument('--probing_num_workers', type=int, default=1)
    parser.add_argument('--num_batches_per_epoch', type=int, default=100)
    parser.add_argument('--val_every_n_epoch', type=int, default=10)

    parser.add_argument('--base_channels', type=int, default=32)  # BASE_CHANNELS
    parser.add_argument('--num_scales', type=int, default=5)  # NUM_SCALES

    return parser.parse_args()


def main(args):

    # Task.init(
    #     project_name='Label',
    #     task_name='pretraining_nako1000_separateinfonce_32x5_50000'
    # )

    patch_size = tuple(args.patch_size)

    pretrain_dataset = PretrainingDataset(
        patch_size=patch_size,
        max_num_voxels_per_patch=MAX_NUM_VOXELS_PER_PATCH,
        batch_size=args.pretrain_batch_size,
        pretraining_dataset=args.pretraining_dataset
    )

    pretrain_dataloader = DataLoader(
        dataset=pretrain_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=args.pretrain_num_workers
    )

    in_channels = 1
    backbone = FPN3d(in_channels, args.base_channels, args.num_scales)
    model = Vox2Vec(
        backbone=backbone,
        base_channels=args.base_channels,
        num_scales=args.num_scales,
    )

    # online probing
    # if args.probing_dataset == 'btcv':
    #     probing_datamodule = BTCV(
    #         root=args.probing_data_dir,
    #         cache_dir=args.cache_dir,
    #         spacing=spacing,
    #         window_hu=WINDOW_HU,
    #         patch_size=patch_size,
    #         batch_size=args.probing_batch_size,
    #         num_batches_per_epoch=args.num_batches_per_epoch,
    #         num_workers=args.probing_num_workers,
    #         buffer_size=100,
    #         split=0
    #     )
    #     num_classes = BTCV.num_classes
    # elif args.probing_dataset == 'amos_mri':
    #     probing_datamodule = AMOSMRI(
    #         root=args.probing_data_dir,
    #         cache_dir=args.cache_dir,
    #         patch_size=patch_size,
    #         batch_size=args.probing_batch_size,
    #         num_batches_per_epoch=args.num_batches_per_epoch,
    #         num_workers=args.probing_num_workers,
    #         buffer_size=100,
    #         split=0
    #     )
    #     num_classes = AMOSMRI.num_classes
    # else:
    #     raise NotImplementedError(f'Dataset {args.dataset} is not supported yet.')
    #
    # heads = [
    #     FPNLinearHead(args.base_channels, args.num_scales, num_classes),
    #     FPNNonLinearHead(args.base_channels, args.num_scales, num_classes)
    # ]
    # probing_callback = OnlineProbing(*heads, patch_size=patch_size)

    checkpoint_callback_1 = ModelCheckpoint(every_n_epochs=25)
    checkpoint_callback_2 = ModelCheckpoint(save_top_k=4, monitor='epoch', mode='max', every_n_epochs=250, filename='{epoch:02d}')

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=args.log_dir, name='pretrain/'),
        # callbacks=[checkpoint_callback, probing_callback],
        callbacks=[checkpoint_callback_1, checkpoint_callback_2],
        accelerator='gpu',
        max_epochs=500,
        gradient_clip_val=1.0
    )
    trainer.fit(
        model=model,
        train_dataloaders={
            'pretrain': pretrain_dataloader,
            # 'online_probing': probing_datamodule.train_dataloader()
        },
        # val_dataloaders=probing_datamodule.val_dataloader(),
    )
    # probing_datamodule.train_dataset.pipeline.close()

if __name__ == '__main__':
    main(parse_args())
