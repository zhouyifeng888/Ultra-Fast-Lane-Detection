
import mindspore.nn as nn
import mindspore.ops as P

class SoftmaxFocalLoss(nn.Cell):
    def __init__(self, gamma, ignore_lb=255, num_lanes=4):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.softmax=P.Softmax(axis=1)
        self.nll = P.NLLLoss(ignore_index=ignore_lb)

    def construct(self, logits, labels):
        scores = self.softmax(logits)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class ParsingRelationLoss(nn.Cell):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()
    def construct(self,logits):
        n,c,h,w = logits.shape
        loss_all = []
        for i in range(0,h-1):
            loss_all.append(logits[:,:,i,:] - logits[:,:,i+1,:])
        #loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss,torch.zeros_like(loss))



class ParsingRelationDis(nn.Cell):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
    def construct(self, x):
        n,dim,num_rows,num_cols = x.shape
        x = torch.nn.functional.softmax(x[:,:dim-1,:,:],dim=1)
        embedding = torch.Tensor(np.arange(dim-1)).float().to(x.device).view(1,-1,1,1)
        pos = torch.sum(x*embedding,dim = 1)

        diff_list1 = []
        for i in range(0,num_rows // 2):
            diff_list1.append(pos[:,i,:] - pos[:,i+1,:])

        loss = 0
        for i in range(len(diff_list1)-1):
            loss += self.l1(diff_list1[i],diff_list1[i+1])
        loss /= len(diff_list1) - 1
        return loss
