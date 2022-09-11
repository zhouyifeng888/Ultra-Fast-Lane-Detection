

import numpy as np

import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor


class SoftmaxFocalLoss(nn.Cell):
    def __init__(self, gamma, num_lanes=4):
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
        loss = self.nll(log_score, labels, self.weight)
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
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()

    def construct(self, x):
        n, dim, num_rows, num_cols = x.shape
        x = torch.nn.functional.softmax(x[:, :dim - 1, :, :], dim=1)
        embedding = torch.Tensor(
            np.arange(dim - 1)).float().to(x.device).view(1, -1, 1, 1)
        pos = torch.sum(x * embedding, dim=1)

        diff_list1 = []
        for i in range(0, num_rows // 2):
            diff_list1.append(pos[:, i, :] - pos[:, i + 1, :])

        loss = 0
        for i in range(len(diff_list1) - 1):
            loss += self.l1(diff_list1[i], diff_list1[i + 1])
        loss /= len(diff_list1) - 1
        return loss
