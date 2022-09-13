
import os
import json

from PIL import Image
import numpy as np

from src.config import config as cfg

data_root = '../../dataset/Tusimple/train_set'
train_gt_path = os.path.join(data_root, 'train_gt.txt')
label_info_path1 = os.path.join(data_root, 'label_data_0313.json')
label_info_path2 = os.path.join(data_root, 'label_data_0531.json')
label_info_path3 = os.path.join(data_root, 'label_data_0601.json')

label_info_lines = []
with open(label_info_path1) as f:
    lines = f.readlines()
    label_info_lines.extend(lines)
with open(label_info_path2) as f:
    lines = f.readlines()
    label_info_lines.extend(lines)
with open(label_info_path3) as f:
    lines = f.readlines()
    label_info_lines.extend(lines)
label_info_dict = {}
for line in label_info_lines:
    json_data = json.loads(line)
    label_info_dict[json_data['raw_file']]=json_data

with open(train_gt_path) as f:
    lines = f.readlines()
    
for i in range(len(lines)):
    line = lines[i]
    infos = line.split()
    img_file_path = infos[0]
    label_file_path = os.path.join(data_root, infos[1])
    
    img = Image.open(os.path.join(data_root,img_file_path))
    w, h = img.size
    png_arr = np.zeros((h, w))
    
    label_info = label_info_dict[img_file_path]
    for lane_id in range(len(label_info['lanes'])):
        label_id = lane_id+1
        lane = label_info['lanes'][lane_id]
        for coordinate_id in range(len(lane)):
            lane_w = lane[coordinate_id]
            lane_h = label_info['h_samples'][coordinate_id]
            if label_id<=cfg.num_lanes and lane_w!=-2:
                w_l = int(lane_w-(w/cfg.griding_num)/2)
                w_r = int(lane_w+(w/cfg.griding_num)/2)
                h_t = int(lane_h-(h/len(cfg.row_anchor))/2)
                h_b = int(lane_h+(h/len(cfg.row_anchor))/2)
                png_arr[h_t:h_b, w_l:w_r] = label_id
            
    png_label = Image.fromarray(png_arr.astype(np.uint8))
    png_label.save(label_file_path)
    
    print(f'finished {i+1} {line}')
        

