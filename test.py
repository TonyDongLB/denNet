import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

print('torch.__version__ {}'.format(torch.__version__))


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def from2d(a):
    a = a.contiguous().view(a.size(0), a.size(1), -1)
    a = a.transpose(1, 2)
    a = a.contiguous().view(-1, a.size(2)).squeeze()
    return a


def dice_loss(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target.numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)  # b,c,h
    num = torch.sum(num, dim=2)

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2 * (num / (den1 + den2))
    dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total


n = 2
c = 2
h = 5
w = 4
criterion = nn.NLLLoss()
# a = t.rand(1, 2, 5, 4).float()
# print(a)
# a = a.contiguous().view(a.size(0), a.size(1), -1)
# a = a.transpose(1, 2)
# a = a.contiguous().view(-1, a.size(2)).squeeze()
# print(a)
# a = torch.ones(1,1,h,w)
# b = torch.zeros(1,1,h,w)
# inputs = torch.cat((b,a),dim=1).float()
# targets = a.long()
# inputs = from2d(inputs)
# targets = from2d(targets)
# inputs = F.log_softmax(inputs)
# print(inputs)
# loss = criterion(inputs, targets)
# print(loss)



mask_pred = Variable(t.rand(1, 2, 5, 4))
print(mask_pred)
height = mask_pred.size(2)
width = mask_pred.size(3)
mask_pred = mask_pred.contiguous().view(mask_pred.size(0),mask_pred.size(1), -1)
mask_pred = mask_pred.transpose(1, 2)
mask_pred = mask_pred.contiguous().view(-1, mask_pred.size(2)).squeeze()
mask_pred = torch.argmax(F.log_softmax(mask_pred,  dim=1), dim=1)
mask_pred = mask_pred.reshape(height, width).float()
print(mask_pred)

#
# a = torch.ones(1, 1,h,w).float()
# b = torch.zeros(1,1,h,w).float()
# print(((a - b) > 1).float())
# inputs = torch.cat((b,a),dim=1).float()
# targets = inputs
# print(dice_loss(inputs, targets))
# print(inputs)
# print(targets)
# # print(inp.size())
# # inputs = torch.rand(n, c, h, w)
# # targets = torch.LongTensor(n, 1, h, w).random_(c)
# inputs_fl = Variable(inputs.clone(), requires_grad=True)
# targets_fl = Variable(targets.clone())
# FL = FocalLoss2d()
# loss = FL(inputs_fl, targets_fl)
# print(loss)
