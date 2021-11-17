print('print importing lib...')

import argparse
from azureml.core import Run
from azureml.core import Dataset
import shutil
import os
import numpy as np
import pandas as pd
import random
import Augmentor

print("lib imported...")

#get scripts arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image-folder', type=str, dest='image_folder')
parser.add_argument('--mask-folder', type=str, dest='mask_folder')
parser.add_argument('--augmented_data', type=str, dest='augmented_data')
args = parser.parse_args()

save_folder = args.augmented_data

run = Run.get_context()

#load data
print('data...')
mask_path = run.input_datasets['mask']
image_path = run.input_datasets['image']

#
os.makedirs(save_folder, exist_ok=True)

#move picture to intermedieate container for data augmentation
image_dir = image_path
mask_dir = mask_path
mask = '_labelIds'

new_image_dir= os.path.join(save_folder, 'image')
new_mask_dir= os.path.join(save_folder, 'mask')
os.makedirs(new_image_dir, exist_ok=True)
os.makedirs(new_mask_dir, exist_ok=True)
reduction = '_leftImg8bit'

#extract list of pictures and rename mask and picture for augmentor ground truth
print('copy file to container')
image_list = []
for root, dirs, files in os.walk(image_dir):
    for name in files:        
        old_path = os.path.join(root, name)
        new_name = name.replace('_leftImg8bit', '')        
        new_path = os.path.join(new_image_dir, new_name)
        if not os.path.exists(new_path):
            shutil.copy(old_path, new_path)
        
        image_list.append(new_path)
        

mask_list = []
for root, dirs, files in os.walk(mask_dir):
    for name in files:
        if mask in name:
            old_path = os.path.join(root, name)
            new_name = name.replace('_gtFine_labelIds', '')        
            new_path = os.path.join(new_mask_dir, new_name)
            if not os.path.exists(new_path):
                shutil.copy(old_path, new_path)

            mask_list.append(new_path)

print('build augmentor pipeline')
            
#build augmentor pipeline
p = Augmentor.Pipeline(new_image_dir)

#add groud truth for pair modification
p.ground_truth(new_mask_dir)

#add operation
p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.skew(probability=0.5, magnitude=0.5)
p.skew_tilt(probability=0.5, magnitude=0.5)
p.random_distortion(probability=0.5,grid_height=4, grid_width=4, magnitude=4)
p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
p.gaussian_distortion(probability=0.5, corner='bell', method='in', grid_height=4, grid_width=4, magnitude=4)
p.skew_top_bottom(probability=0.5, magnitude=.5)
p.skew_left_right(probability=0.5, magnitude=.5)
p.skew_corner(probability=0.5, magnitude=.5)
p.resize(probability=1,width=256, height=256)

print('start sampling')
p.sample(300)

run.complete()
