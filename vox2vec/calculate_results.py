import os
import json
import pandas as pd

ROOT_DIR = '/home/kats/storage/staff/eytankats/experiments/label/vox2vec/submission/results_csv/amos_mri_fine_tuning_equal_contrib_1convhead_32x5_50000_4'
FOLDS = [0, 1, 2, 3, 4]

df = pd.DataFrame()
for fold in FOLDS:

    results_file = os.path.join(ROOT_DIR, f'test_metrics_{fold}.json')
    with open(results_file) as f:
        d = json.load(f)

        d[0]['test/head_0_dice_score_for_cls_1'] = (d[0]['test/head_0_dice_score_for_cls_1'] + d[0]['test/head_0_dice_score_for_cls_2']) / 2
        del d[0]['test/head_0_dice_score_for_cls_2']

        d[0]['test/head_0_dice_score_for_cls_10'] = (d[0]['test/head_0_dice_score_for_cls_10'] + d[0]['test/head_0_dice_score_for_cls_11']) / 2
        del d[0]['test/head_0_dice_score_for_cls_11']

        fold_df = pd.DataFrame(d)
        df = pd.concat([df, fold_df], ignore_index=True)

df.loc['mean'] = df.mean()
df.loc['std'] = df.std()

csv_path = os.path.join(ROOT_DIR, f'statistics.csv')
df.to_csv(csv_path)
