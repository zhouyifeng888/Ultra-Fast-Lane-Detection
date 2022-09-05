
import os

from PIL import Image
import numpy as np

data_root = '../../dataset/Tusimple'
train_gt_path = os.path.join(data_root, 'train_gt.txt')

with open(train_gt_path) as f:
    lines = f.readlines()
    
for line in lines:
    infos = line.split()
    img_file_path = infos[0]
    label_file_path = infos[1]

