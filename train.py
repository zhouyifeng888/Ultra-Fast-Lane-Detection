
import numpy as np

from mindspore import context

from src.network import ParsingNet


if __name__ == '__main__':
    
    from mindspore import Tensor
    import mindspore as ms
    
    context.set_context(mode=context.GRAPH_MODE,  device_target='GPU')
    
    input_shape = (1, 3, 288, 800)
    t_data = Tensor(np.ones(input_shape), ms.float32)
    
    net = ParsingNet('18', '../resnet18_ascend_v150_imagenet2012_official_cv_top1acc70.47_top5acc89.61.ckpt')
    result = net(t_data)
    print(f'result.shape:{result.shape}')