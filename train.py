
import time

import numpy as np

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.resnet import get_resnet
from src.network import ParsingNet
from src.utils import print_trainable_params_count
from src.dataset import create_lane_dataset


def main():
    device_id = int(os.getenv('DEVICE_ID', 0))
    device_num = int(os.getenv('RANK_SIZE', 1))
    print(f'device_id:{device_id}')
    print(f'device_num:{device_num}')
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target)
    context.set_context(device_id=device_id)
    context.set_context(enable_graph_kernel=True)

    #profiler = Profiler()

    if args.start_epochs < 1 or args.start_epochs > args.epochs:
        print(f'start_epochs must between 1 and {args.epochs}')
        return

    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    if args.resume.startswith('s3://') or args.resume.startswith('obs://'):
        local_resume = os.path.join(
            args.local_data_root, f'resume_{device_id}.ckpt')
        file.copy_parallel(args.resume, local_resume)
        args.resume = local_resume

    data_type = ms.float32 if args.amp_level == 'O0' else ms.float16
    use_kaiming_uniform = True if args.use_kaiming_uniform == 'True' else False
    if args.arch == 'birealnet34':
        net = birealnet34(args.num_classes, data_type=data_type,
                          use_kaiming_uniform=use_kaiming_uniform)
    elif args.arch == 'birealnet18':
        net = birealnet18(args.num_classes, data_type=data_type,
                          use_kaiming_uniform=use_kaiming_uniform)

    print_trainable_params_count(net)

    if args.resume and args.resume != 'None':
        ckpt = load_checkpoint(args.resume)
        load_param_into_net(net, ckpt)
        print('load ckpt success')

    copy_result_file = os.path.join(
        args.local_data_root, 'local_data_path.txt')
    if device_num == 1 or device_id == 0:
        if args.data_url.startswith('s3://') or args.data_url.startswith('obs://'):
            start = time.time()
            print('start copy data...')
            local_data_path = os.path.join(args.local_data_root, 'data')
            file.copy_parallel(args.data_url, local_data_path)
            end = time.time()
            print(f'copy data finished,use time{end-start}s')
        else:
            local_data_path = args.data_url

        if args.data_format == 'tar.gz':
            start = time.time()
            print('start decompression imagenet tar.gz...')
            os.chdir(local_data_path)
            os.system('cat imagenet.tar.gza*|tar -zx')
            local_data_path = os.path.join(
                local_data_path, 'imagenet_original')
            end = time.time()
            print(
                f'decompression imagenet tar.gz finished,use time{end-start}s')
        with open(copy_result_file, 'w') as f:
            f.write(local_data_path)

    if device_num > 1 and device_id != 0:
        local_data_path = None
        while local_data_path is None:
            time.sleep(5)
            if os.path.exists(copy_result_file):
                with open(copy_result_file) as f:
                    local_data_path = f.readline()

    train_path = os.path.join(local_data_path, 'imagenet/train')
    train_dataset = create_imagenet_dataset(
        'train', train_path, args.image_size, args.batch_size, rank=device_id, group_size=device_num)
    val_path = os.path.join(local_data_path, 'imagenet/val')
    val_dataset = create_imagenet_dataset(
        'eval', val_path, args.image_size, args.batch_size)

    batches_per_epoch = train_dataset.get_dataset_size()
    print(f'batches_per_epoch:{batches_per_epoch}')

    # define loss function (criterion) and optimizer
    loss = LabelSmoothingCrossEntropy(
        smooth_factor=args.smoothing, num_classes=args.num_classes)

    if args.use_cosine_decay_lr == 'True':
        lr = warmup_cosine_annealing_lr_V2(lr=args.lr, steps_per_epoch=batches_per_epoch, warmup_epochs=args.warmup,
                                           warmup_init_lr=args.warmup_init_lr,
                                           max_epoch=args.epochs - args.cooldown, T_max=args.epochs, eta_min=args.lr_min,
                                           cooldown_epochs=args.cooldown)
    else:
        lr = []
        for epoch_index in range(args.epochs):
            for _ in range(batches_per_epoch):
                step_lr = (1.0 - epoch_index / args.epochs) * args.lr
                lr.append(step_lr)
        lr = np.array(lr).astype(np.float32)
    args.start_epochs = args.start_epochs - 1
    if args.start_epochs > 0:
        lr = lr[args.start_epochs * batches_per_epoch:]
    lr = Tensor(lr)

    weight_parameters = []
    other_parameters = []
    for param in net.trainable_params():
        if len(param.shape) == 4:
            weight_parameters.append(param)
        else:
            other_parameters.append(param)
    group_params = [{'params': other_parameters},
                    {'params': weight_parameters, 'weight_decay': args.weight_decay},
                    {'order_params': net.trainable_params()}]

    opt = nn.Adam(
        group_params,
        lr,
        weight_decay=args.weight_decay,
        use_nesterov=True,
        loss_scale=1024.0 if args.amp_level == 'O3' else 1.0
    )
    loss_scale_manager = FixedLossScaleManager(
        1024.0, drop_overflow_update=False)
    if args.amp_level == 'O3':
        model = Model(net, loss, opt, metrics={'loss', 'top_1_accuracy', 'top_5_accuracy'},
                      amp_level=args.amp_level, loss_scale_manager=loss_scale_manager)
    else:
        model = Model(net, loss, opt, metrics={'loss', 'top_1_accuracy', 'top_5_accuracy'},
                      amp_level=args.amp_level)

#    if args.resume and args.resume != 'None':
#        ckpt = load_checkpoint(args.resume)
#        load_param_into_net(model._train_network, ckpt)
#        print('load ckpt success')

    loss_cb = LossMonitor(
        per_print_times=1 if args.device_target == "CPU" else batches_per_epoch)
    time_cb = TimeMonitor(data_size=batches_per_epoch)

    callbacks = [time_cb, loss_cb, Val_Callback(
        model, val_dataset, device_id, args.train_url)]

    args.dataset_sink_mode = True if args.dataset_sink_mode == 'True' else False
    model.train(args.epochs - args.start_epochs, train_dataset,
                callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)

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
    
    
    
    