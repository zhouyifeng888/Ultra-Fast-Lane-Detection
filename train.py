
import numpy as np

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.resnet import get_resnet
from src.network import ParsingNet


if __name__ == '__main__':
    
    from mindspore import Tensor
    import mindspore as ms
    
    context.set_context(mode=context.GRAPH_MODE,  device_target='GPU')
    
    input_shape = (1, 3, 288, 800)
    t_data = Tensor(np.ones(input_shape), ms.float32)
    
    backbone = get_resnet()
    param_dict = load_checkpoint('../resnet18_ascend_v150_imagenet2012_official_cv_top1acc70.47_top5acc89.61.ckpt')
    load_param_into_net(backbone, param_dict)
    print('load resnet pretrain ckpt success')
    
    net = ParsingNet('18', backbone)
    result = net(t_data)
    print(f'result.shape:{result.shape}')
    
    
    
    