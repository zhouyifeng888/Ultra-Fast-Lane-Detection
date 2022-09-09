
import numpy as np

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.resnet import get_resnet
from src.network import ParsingNet
from src.utils import print_trainable_params_count
from src.dataset import create_lane_dataset


if __name__ == '__main__':
    
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
    
    dataset = create_lane_dataset(
        '../../dataset/Tusimple/train_set/', 'train_gt.txt', 16, num_workers=8)
    data = next(dataset.create_dict_iterator())
    
    
    
    