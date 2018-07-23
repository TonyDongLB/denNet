import torch
import torch.nn.functional as F
from torch.autograd import Variable


from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    tot = 0
    for i, (imgs, true_masks) in enumerate(dataset):
        imgs = Variable(imgs)
        true_masks = Variable(true_masks)
        if gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

        mask_pred = net(imgs)[0]
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

        tot += dice_coeff(mask_pred, true_masks)[0]
    return tot / i
