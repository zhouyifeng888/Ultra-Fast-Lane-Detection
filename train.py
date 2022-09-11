
"""train the Ultra-Fast-Lane-Detection model"""
try:
    from moxing.framework import file
    print("import moxing success")
except ModuleNotFoundError as e:
    print(f'not modelarts env, error={e}')

import os
import time

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.train.callback._time_monitor import TimeMonitor
from mindspore.train.callback._loss_monitor import LossMonitor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication import init
from mindspore.train.model import ParallelMode
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from src.resnet import get_resnet
from src.network import ParsingNet
from src.utils import print_trainable_params_count
from src.dataset import create_lane_dataset
from src.config import config as cfg
from src.loss import TrainLoss, NetWithLossCell
from src.lr_scheduler import warmup_cosine_annealing_lr_V2


def main():
    device_id = int(os.getenv('DEVICE_ID', 0))
    device_num = int(os.getenv('RANK_SIZE', 1))
    print(f'device_id:{device_id}')
    print(f'device_num:{device_num}')
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target)
    context.set_context(device_id=device_id)
#    context.set_context(enable_graph_kernel=True)

    #profiler = Profiler()

    if cfg.start_epochs < 1 or cfg.start_epochs > cfg.epochs:
        print(f'start_epochs must between 1 and {cfg.epochs}')
        return

    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    if cfg.resume.startswith('s3://') or cfg.resume.startswith('obs://'):
        local_resume = os.path.join(
            cfg.local_data_root, f'resume_{device_id}.ckpt')
        file.copy_parallel(cfg.resume, local_resume)
        cfg.resume = local_resume

    if cfg.backbone_pretrain.startswith('s3://') or cfg.backbone_pretrain.startswith('obs://'):
        local_backbone_pretrain = os.path.join(
            cfg.local_data_root, f'backbone_pretrain_{device_id}.ckpt')
        file.copy_parallel(cfg.backbone_pretrain, local_backbone_pretrain)
        cfg.backbone_pretrain = local_backbone_pretrain

    backbone = get_resnet(resnet_type=cfg.backbone)
    if cfg.backbone_pretrain and cfg.backbone_pretrain != 'None':
        param_dict = load_checkpoint(cfg.backbone_pretrain)
        load_param_into_net(backbone, param_dict)
        print('load resnet pretrain ckpt success')

    net = ParsingNet(cfg.backbone, backbone, cls_dim=(
        cfg.griding_num + 1, len(cfg.row_anchor), cfg.num_lanes), use_aux=True)

    print_trainable_params_count(net)

    if cfg.resume and cfg.resume != 'None':
        ckpt = load_checkpoint(cfg.resume)
        load_param_into_net(net, ckpt)
        print('load ckpt success')

    copy_result_file = os.path.join(
        cfg.local_data_root, 'local_data_path.txt')
    if device_num == 1 or device_id == 0:
        if cfg.data_url.startswith('s3://') or cfg.data_url.startswith('obs://'):
            start = time.time()
            print('start copy data...')
            local_data_path = os.path.join(cfg.local_data_root, 'data')
            file.copy_parallel(cfg.data_url, local_data_path)
            end = time.time()
            print(f'copy data finished,use time{end-start}s')
        else:
            local_data_path = cfg.data_url

        with open(copy_result_file, 'w') as f:
            f.write(local_data_path)

    if device_num > 1 and device_id != 0:
        local_data_path = None
        while local_data_path is None:
            time.sleep(5)
            if os.path.exists(copy_result_file):
                with open(copy_result_file) as f:
                    local_data_path = f.readline()

    train_dataset = create_lane_dataset(local_data_path, 'train_gt.txt', cfg.batch_size,
                                        rank_size=device_num, rank_id=device_id)

#    train_path = os.path.join(local_data_path, 'imagenet/train')
#    train_dataset = create_imagenet_dataset(
#        'train', train_path, args.image_size, args.batch_size, rank=device_id, group_size=device_num)
#    val_path = os.path.join(local_data_path, 'imagenet/val')
#    val_dataset = create_imagenet_dataset(
#        'eval', val_path, args.image_size, args.batch_size)

    batches_per_epoch = train_dataset.get_dataset_size()
    print(f'batches_per_epoch:{batches_per_epoch}')

    data_type = ms.float32 if cfg.amp_level == 'O0' else ms.float16
    loss_fn = TrainLoss(data_type=data_type)
    net_with_loss = NetWithLossCell(net, loss_fn)

    if cfg.use_cosine_decay_lr == 'True':
        lr = warmup_cosine_annealing_lr_V2(lr=cfg.lr, steps_per_epoch=batches_per_epoch, warmup_epochs=cfg.warmup,
                                           warmup_init_lr=cfg.warmup_init_lr,
                                           max_epoch=cfg.epochs - cfg.cooldown, T_max=cfg.epochs, eta_min=cfg.lr_min,
                                           cooldown_epochs=cfg.cooldown)
    else:
        lr = []
        for epoch_index in range(cfg.epochs):
            for _ in range(batches_per_epoch):
                step_lr = (1.0 - epoch_index / cfg.epochs) * cfg.lr
                lr.append(step_lr)
        lr = np.array(lr).astype(np.float32)
    cfg.start_epochs = cfg.start_epochs - 1
    if cfg.start_epochs > 0:
        lr = lr[cfg.start_epochs * batches_per_epoch:]
    lr = Tensor(lr)

    if cfg.optimizer == 'Adam':
        opt = nn.Adam(
            net.trainable_params(),
            lr,
            weight_decay=cfg.weight_decay,
            use_nesterov=True,
            loss_scale=1024.0 if cfg.amp_level == 'O3' else 1.0
        )
    else:
        opt = nn.SGD(
            net.trainable_params(),
            lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
            loss_scale=1024.0 if cfg.amp_level == 'O3' else 1.0
        )
    loss_scale_manager = FixedLossScaleManager(
        1024.0, drop_overflow_update=False)
    if cfg.amp_level == 'O3':
        model = Model(net_with_loss, None, opt, boost_level='O0',
                      amp_level=cfg.amp_level, loss_scale_manager=loss_scale_manager)
    else:
        model = Model(net_with_loss, None, opt, boost_level='O0',
                      amp_level=cfg.amp_level)

    loss_cb = LossMonitor(per_print_times=1)
    time_cb = TimeMonitor(data_size=batches_per_epoch)

    callbacks = [time_cb, loss_cb]

    cfg.dataset_sink_mode = True if cfg.dataset_sink_mode == 'True' else False
    model.train(cfg.epochs - cfg.start_epochs, train_dataset,
                callbacks=callbacks, dataset_sink_mode=cfg.dataset_sink_mode)

    # profiler.analyse()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f'Total time: {duration}s.', duration)

#    from mindspore import Tensor
#    import mindspore as ms
#
#    context.set_context(mode=context.GRAPH_MODE,  device_target='GPU')
#
#    input_shape = (1, 3, 288, 800)
#    t_data = Tensor(np.ones(input_shape), ms.float32)
#
#    backbone = get_resnet()
#    param_dict = load_checkpoint('../resnet18_ascend_v150_imagenet2012_official_cv_top1acc70.47_top5acc89.61.ckpt')
#    load_param_into_net(backbone, param_dict)
#    print('load resnet pretrain ckpt success')
#
#    griding_num=200
#    cls_num_per_lane=18
#    num_lanes=4
#    net_train = ParsingNet('18', backbone, cls_dim=(griding_num+1,cls_num_per_lane, num_lanes),use_aux=True)
#    net_train.set_train(True)
#    print_trainable_params_count(net_train)
#    result = net_train(t_data)
#    print(f'net_train result[0].shape:{result[0].shape},result[1].shape:{result[1].shape}')
#
#    net_eval = ParsingNet('18', backbone, cls_dim=(griding_num+1,cls_num_per_lane, num_lanes), use_aux=False)
#    net_eval.set_train(False)
#    print_trainable_params_count(net_eval)
#    result = net_eval(t_data)
#    print(f'net_eval result.shape:{result.shape}')

#    dataset = create_lane_dataset(
#        '../../dataset/Tusimple/train_set/', 'train_gt.txt', 16, num_workers=8)
#    data = next(dataset.create_dict_iterator())
