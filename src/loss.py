

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor

from src.config import config as cfg


class SoftmaxFocalLoss(nn.Cell):
    def __init__(self, gamma=2, num_lanes=4):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.softmax = P.Softmax(axis=1)
        self.pow = P.Pow()
        self.log_softmax = nn.LogSoftmax(axis=1)
        self.weight = Tensor(np.ones((num_lanes,))).astype(np.float32)
        self.nll = P.NLLLoss(reduction="mean")

    def construct(self, logits, labels):
        scores = self.softmax(logits)
        factor = self.pow(1.0 - scores, self.gamma)
        log_score = self.log_softmax(logits)
        log_score = factor * log_score
        print(f'log_score.shape :{log_score.shape}============')
        print(f'labels.shape :{labels.shape}==============')
        loss = self.nll(log_score, labels, self.weight)
        print(f'loss.shape :{loss.shape}======================')
        return loss


class ParsingRelationLoss(nn.Cell):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()
        self.concat = P.Concat(axis=0)
        self.zeros_like = P.ZerosLike()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=1.0, reduction='mean')

    def construct(self, logits):
        n, c, h, w = logits.shape
        loss_all = []
        for i in range(0, h - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        loss = self.concat(loss_all)
        return self.smooth_l1_loss(loss, self.zeros_like(loss))


class ParsingRelationDis(nn.Cell):
    def __init__(self, griding_num=100, anchor_nums=56, num_lanes=4, data_type=ms.float16):
        super(ParsingRelationDis, self).__init__()
        self.dim = griding_num
        self.num_rows = anchor_nums
        self.num_cols = num_lanes

        self.softmax = P.Softmax(axis=1)
        self.embedding = Tensor(
            np.arange(griding_num)).astype(data_type).view((1, -1, 1, 1))
        self.reduce_sum = P.ReduceSum(keep_dims=False)

        self.l1_loss = nn.L1Loss(reduction='mean')

    def construct(self, x):
        x = self.softmax(x[:, :self.dim, :, :])
        pos = self.reduce_sum(x * self.embedding, 1)
        diff_list1 = []
        for i in range(0, self.num_rows // 2):
            diff_list1.append(pos[:, i, :] - pos[:, i + 1, :])

        loss = 0
        for i in range(len(diff_list1) - 1):
            loss += self.l1_loss(diff_list1[i], diff_list1[i + 1])
        loss = loss / (len(diff_list1) - 1)
        return loss


class TrainLoss(nn.Cell):
    def __init__(self, gamma=2, data_type=ms.float16):
        super(TrainLoss, self).__init__()
        self.w1 = 1.0
        self.loss1 = SoftmaxFocalLoss(gamma=gamma, num_lanes=cfg.num_lanes)
        self.w2 = cfg.sim_loss_w
        self.loss2 = ParsingRelationLoss()
        self.w3 = 1.0
        self.loss3 = nn.SoftmaxCrossEntropyWithLogits(
            sparse=False, reduction='mean')
        self.w4 = cfg.shp_loss_w
        self.loss4 = ParsingRelationDis(
            griding_num=cfg.griding_num, anchor_nums=len(cfg.row_anchor), num_lanes=cfg.num_lanes, data_type=data_type)

    def construct(self, cls_out, seg_out, cls_label, seg_label):
        total_loss = self.w1 * self.loss1(cls_out, cls_label) + self.w2 * self.loss2(
            cls_out) + self.w3 * self.loss3(seg_out, seg_label) + self.w4 * self.loss4(cls_out)
        return total_loss


class NetWithLossCell(nn.Cell):
    def __init__(self, network, loss_fn):
        super(NetWithLossCell, self).__init__()
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, x, cls_label, seg_label):
        cls_out, seg_out = self.network(x)
        loss = self.loss_fn(cls_out, seg_out, cls_label, seg_label)
        return loss
