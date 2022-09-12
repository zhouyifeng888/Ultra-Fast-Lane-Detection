
"""eval the Ultra-Fast-Lane-Detection model"""
import os
import time
import json

import numpy as np

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.resnet import get_resnet
from src.network import ParsingNet
from src.utils import print_trainable_params_count, TusimpleAccEval
from src.dataset import create_lane_test_dataset
from src.config import config as cfg


def main():
    device_id = int(os.getenv('DEVICE_ID', 0))
    device_num = int(os.getenv('RANK_SIZE', 1))
    print(f'device_id:{device_id}')
    print(f'device_num:{device_num}')
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target)
    context.set_context(device_id=device_id)
#    context.set_context(enable_graph_kernel=True)

    backbone = get_resnet(resnet_type=cfg.backbone)
    net = ParsingNet(cfg.backbone, backbone, cls_dim=(
        cfg.griding_num + 1, len(cfg.row_anchor), cfg.num_lanes), use_aux=True)
    print_trainable_params_count(net)

    if cfg.resume and cfg.resume != 'None':
        ckpt = load_checkpoint(cfg.resume)
        load_param_into_net(net, ckpt)
        print('load ckpt success')

    with open(os.path.join(cfg.data_url, 'test_set', 'test_label.json')) as f:
        label_lines = f.readlines()
    label_info_list = []
    for i in range(len(label_lines)):
        json_data = json.loads(label_lines[i])
        label_info_list.append(json_data)

    val_dataset = create_lane_test_dataset(
        os.path.join(cfg.data_url, 'test_set'), 'test_label.json', cfg.batch_size)

    accEval = TusimpleAccEval
    acc = 0
    total_count = 0
    for data in val_dataset.create_dict_iterator():
        imgs = data['image']
        batch_index = data['index'].asnumpy()
        results = net(imgs).asnumpy()
        for i in range(results.shape[0]):
            index = batch_index[i]
            gt_lanes = np.array(label_info_list[index]['lanes'])
            y_samples = np.array(label_info_list[index]['h_samples'])

            pred_one_img_lanes = accEval.generate_tusimple_lines(
                results[i], imgs[0, 0].shape, cfg.griding_num)
            one_img_acc = accEval.bench(pred_one_img_lanes,
                                        gt_lanes, y_samples)
            acc += one_img_acc
            total_count += 1
            if total_count%100==0:
                print(f'total_count:{total_count}')
    acc = acc / total_count
    print(f'accuracy:{acc}')


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f'Total time: {duration}s.', duration)
