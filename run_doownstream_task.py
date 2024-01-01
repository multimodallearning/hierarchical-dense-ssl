# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning

from clearml import Task

from vox2vec.default_params import *
from vox2vec.eval.downstream_dataset import DownstreamDataset
from vox2vec.nn import FPN3d, FPNLinearHead, FPNNonLinearHead
from vox2vec.eval.end_to_end import EndToEnd
from vox2vec.eval.probing import Probing
from vox2vec.utils.misc import save_json

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('--dataset', default='btcv')  # amos or btcv
    parser.add_argument('--setup', default='from_scratch')  # from_scratch, fine-tuning, probing

    parser.add_argument('--ckpt', default='/path/to/pretrained.ckpt')
    parser.add_argument('--log_dir', default='/path/to/output_dir/')

    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--examples_num', default='all')

    parser.add_argument('--patch_size', nargs='+', type=int, default=PATCH_SIZE)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_batches_per_epoch', type=int, default=300)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--warmup_epochs', type=int, default=50)  # used only in finetuning setup

    parser.add_argument('--base_channels', type=int, default=BASE_CHANNELS)
    parser.add_argument('--num_scales', type=int, default=NUM_SCALES)

    return parser.parse_args()

def main(args):

    # Task.init(
    #     project_name='Label',
    #     task_name='amos_mri_fine_tuning_equal_contribution_reproduction_1convhead_32x5_50000_split4'
    # )

    train_dataset = DownstreamDataset(
        dataset=args.dataset,
        patch_size=tuple(args.patch_size),
        batch_size=args.batch_size,
        batches_per_epoch=args.num_batches_per_epoch,
        split=args.split,
        mode='training'
    )
    num_classes = train_dataset.num_classes

    val_dataset = DownstreamDataset(
        dataset=args.dataset,
        patch_size=tuple(args.patch_size),
        batch_size=args.batch_size,
        batches_per_epoch=args.num_batches_per_epoch,
        split=args.split,
        mode='validation'
    )

    test_dataset = DownstreamDataset(
        dataset=args.dataset,
        patch_size=tuple(args.patch_size),
        batch_size=args.batch_size,
        batches_per_epoch=args.num_batches_per_epoch,
        split=args.split,
        mode='test'
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    in_channels = 1
    backbone = FPN3d(in_channels, args.base_channels, args.num_scales)
    if args.setup == 'from_scratch':
        head = FPNLinearHead(args.base_channels, args.num_scales, num_classes)
        model = EndToEnd(backbone, head, patch_size=tuple(args.patch_size))
        callbacks = [
            ModelCheckpoint(save_top_k=1, monitor='val/head_0_avg_dice_score', filename='best_avg', mode='max'),
            ModelCheckpoint(save_top_k=1, monitor='val/head_0_smooth_dice_score', filename='best_smooth', mode='max'),
            ModelCheckpoint(save_top_k=1, filename='last'),
        ]
    elif args.setup == 'probing':
        if args.ckpt is not None:

            state_dict = torch.load(args.ckpt)['state_dict']
            # for key in list(state_dict.keys()):
            #     if 'backbone' not in key:
            #         del state_dict[key]
            # for key in list(state_dict.keys()):
            #     state_dict[key.replace("backbone.", "")] = state_dict.pop(key)
            backbone.load_state_dict(state_dict)

        heads = [
            FPNLinearHead(args.base_channels, args.num_scales, num_classes),
            FPNNonLinearHead(args.base_channels, args.num_scales, num_classes)
        ]
        model = Probing(backbone, *heads, patch_size=tuple(args.patch_size))
        callbacks = [
            ModelCheckpoint(save_top_k=1, monitor='val/head_0_avg_dice_score', filename='best_avg', mode='max'),
            ModelCheckpoint(save_top_k=1, monitor='val/head_0_smooth_dice_score', filename='best_smooth', mode='max'),
            ModelCheckpoint(save_top_k=1, filename='last'),
        ]
    elif args.setup == 'fine-tuning':

        if args.ckpt is not None:
            state_dict = torch.load(args.ckpt)['state_dict']
            # for key in list(state_dict.keys()):
            #     if 'backbone' not in key:
            #         del state_dict[key]
            # for key in list(state_dict.keys()):
            #     state_dict[key.replace("backbone.", "")] = state_dict.pop(key)
            backbone.load_state_dict(state_dict)

        head = FPNLinearHead(args.base_channels, args.num_scales, num_classes)
        model = EndToEnd(backbone, head, patch_size=tuple(args.patch_size))
        callbacks = [
            BackboneFinetuning(unfreeze_backbone_at_epoch=args.warmup_epochs),
            ModelCheckpoint(save_top_k=1, monitor='val/head_0_avg_dice_score', filename='best_avg', mode='max'),
            ModelCheckpoint(save_top_k=1, monitor='val/head_0_smooth_dice_score', filename='best_smooth', mode='max'),
            ModelCheckpoint(save_top_k=1, filename='last'),
        ]
    else:
        raise ValueError(args.setup)

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f'eval/{args.dataset}/{args.setup}/split_{args.split}'
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator='gpu',
        max_epochs=args.max_epochs,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    log_dir = Path(logger.log_dir)
    test_metrics = trainer.test(model, dataloaders=test_dataloader, ckpt_path=log_dir / 'checkpoints/best_avg.ckpt')
    save_json(test_metrics, log_dir / 'test_metrics.json')

    # for split in range(5):
    #     datamodule = AMOSMRI(
    #         root=args.data_root_dir,
    #         cache_dir=args.cache_dir,
    #         patch_size=tuple(args.patch_size),
    #         batch_size=args.batch_size,
    #         num_batches_per_epoch=args.num_batches_per_epoch,
    #         num_workers=args.num_workers,
    #         buffer_size=args.batch_size * 5,
    #         split=split
    #     )
    #
    #     test_metrics = trainer.test(model, datamodule=datamodule, ckpt_path=f'/path/to/downstream.ckpt')
    # df = pd.DataFrame(model.test_results)
    # df.to_csv("/path/to/results.csv", index=False)

if __name__ == '__main__':
    main(parse_args())
