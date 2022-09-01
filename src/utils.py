
import math
import numpy as np

from mindspore.common.initializer import (Normal, One, Uniform, Zero)


def initialize_weight_goog(shape=None, layer_type='conv', bias=False):
    if layer_type not in ('conv', 'bn', 'fc'):
        raise ValueError(
            'The layer type is not known, the supported are conv, bn and fc')
    if bias:
        return Zero()
    if layer_type == 'conv':
        assert isinstance(shape, (tuple, list)) and len(
            shape) == 3, 'The shape must be 3 scalars, and are in_chs, ks, out_chs respectively'
        n = shape[1] * shape[1] * shape[2]
        return Normal(math.sqrt(2.0 / n))
    if layer_type == 'bn':
        return One()
    assert isinstance(shape, (tuple, list)) and len(
        shape) == 2, 'The shape must be 2 scalars, and are in_chs, out_chs respectively'
    n = shape[1]
    init_range = 1.0 / math.sqrt(n)
    return Uniform(init_range)


def print_trainable_params_count(network):
    params = network.trainable_params()
    trainable_params_count = 0
    for i in range(len(params)):
        param = params[i]
        shape = param.data.shape
        size = np.prod(shape)
        trainable_params_count += size
    print("trainable_params_count:" + str(trainable_params_count))
