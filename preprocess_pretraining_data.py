import os
import amid
import random
import pathlib
import subprocess

import numpy as np
import pandas as pd
import nibabel as nib
import reorient_nii as ornt

from matplotlib import pyplot as plt

AMOS_DIR = '/home/kats/storage/staff/eytankats/data/amos/zip/'
FLARE_DIR = '/home/kats/storage/staff/eytankats/data/flare/zip/'
NAKO_DIR = '/home/kats/storage/staff/eytankats/data/nako_1000/nii_allmod/'
C3D_TOOL = '/home/kats/storage/staff/eytankats/tools/c3d-1.1.0-Linux-x86_64/bin/c3d'
OUTPUT_DIR = '/home/kats/storage/staff/eytankats/data/hierarchical_dense_ssl/pretraining/'

dataset = 'nako'  # can be amos, flare, nako
spacing_x = 1.5  # 1.0 for ct 1.5 for mri
spacing_y = 1.5  # 1.0 for ct 1.5 for mri
spacing_z = 1.5  # 2.0 for ct 1.5 for mri
percentile_min = 1  # lower percentile to calculate lower bound for clipping 'mri' images, will be used only if modality == 'mri'
percentile_max = 99.99  # upper percentile to calculate upper bound for clipping 'mri' images, will be used only if modality == 'mri'
vol_clip_min = -175  # -175 is a lower bound for CT abdomen window, will be used only if modality == 'ct'
vol_clip_max = 250  # 250 is an upper bound for CT abdomen window, will be used only if modality == 'ct'

C3D_TASK_STRING_IMG = \
    f"-interpolation Linear " \
    f"-resample-mm {spacing_x}x{spacing_y}x{spacing_z}mm " \

if dataset == 'amos':
    data = amid.AMOS(root=AMOS_DIR)
    ids = data.ids[:500]
elif dataset == 'flare':
    data = amid.FLARE2022(root=FLARE_DIR)
    ids = [data_id for data_id in data.ids if data_id.startswith('TU')]
elif dataset == 'nako':
    ids = list(pathlib.Path(NAKO_DIR).rglob('*2_3D_GRE_TRA_W*.nii.gz'))

output_dir_images = os.path.join(OUTPUT_DIR, dataset, 'images')
output_dir_visualizations = os.path.join(OUTPUT_DIR, dataset, 'visualizations')
output_file_analisys = os.path.join(OUTPUT_DIR, dataset, 'data_analisys.csv')

os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_visualizations, exist_ok=True)

image_names = []
shape_x, shape_y, shape_z = [], [], []
spacings_x, spacings_y, spacings_z = [], [], []
min_val, max_val = [], []
orientation = []
for idx, data_id in enumerate(ids):

    if dataset == 'flare' or dataset == 'amos':
        image_names.append(data_id)
        output_path = pathlib.Path(output_dir_images).joinpath(data_id + '.nii.gz')
        nib.save(nib.Nifti1Image(data.image(data_id), affine=data.affine(data_id)), output_path)
        image_path = output_path
        
    elif dataset == 'nako':
        image_names.append(data_id.name)
        image_path = data_id
        output_path = pathlib.Path(output_dir_images).joinpath(data_id.name)

    args = C3D_TOOL + " " + str(image_path) + " " + C3D_TASK_STRING_IMG + "-o " + str(output_path)
    subprocess.run(args, shell=True)

    nib_image = nib.load(output_path)
    if ornt.get_orientation(nib_image) != 'RAS':
        nib_image = ornt.reorient(nib_image, 'RAS')
    orientation.append(ornt.get_orientation(nib_image))

    np_image = nib_image.get_fdata()
    header = nib_image.header

    shape_x.append(np_image.shape[0])
    shape_y.append(np_image.shape[1])
    shape_z.append(np_image.shape[2])

    spacing = header.get_zooms()
    spacings_x.append(np.round(spacing[0], 1))
    spacings_y.append(np.round(spacing[1], 1))
    spacings_z.append(np.round(spacing[2], 1))

    if dataset == 'flare' or dataset == 'amos':

        np_image = np.clip(np_image, vol_clip_min, vol_clip_max)
        np_image = (np_image - vol_clip_min) / (vol_clip_max - vol_clip_min)
        nib.save(nib.Nifti1Image(np_image.astype(np.float32), affine=nib_image.affine), output_path)

    elif dataset == 'nako':

        vol_clip_min = np.percentile(np_image, percentile_min)
        vol_clip_max = np.percentile(np_image, percentile_max)

        np_image = np.clip(np_image, vol_clip_min, vol_clip_max)
        np_image = (np_image - vol_clip_min) / (vol_clip_max - vol_clip_min)

        nib.save(nib.Nifti1Image(np_image.astype(np.float32), affine=nib_image.affine), output_path)

    min_val.append(np.min(np_image))
    max_val.append(np.max(np_image))

    if ((idx + 1) % 10) == 0:
        print(f'{idx + 1} images processed')

df = pd.DataFrame(data={
    'path': image_names,
    'shape_x': shape_x,
    'shape_y': shape_y,
    'shape_z': shape_z,
    'spacing_x': spacings_x,
    'spacing_y': spacings_y,
    'spacing_z': spacings_z,
    'min_val': min_val,
    'max_val': max_val,
    'orientation': orientation
})

df.to_csv(output_file_analisys)
images_paths = sorted(list(pathlib.Path(output_dir_images).glob('*.nii.gz')))

random_sampled_images_paths = random.choices(images_paths, k=20)
for image_path in random_sampled_images_paths:
    np_image = nib.load(image_path).get_fdata()

    center_slice = np_image.shape[2] // 2
    for image_slice in range(center_slice - 2, center_slice + 3):
        plt.imshow(np_image[:, :, image_slice], cmap='gray')
        plt.colorbar()

        output_path = pathlib.Path(output_dir_visualizations).joinpath(image_path.name[:-7] + '_z' + str(image_slice) + '.png')
        plt.savefig(output_path)
        plt.close()
