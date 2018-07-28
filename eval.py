import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import jaccard_similarity_score as jsc


from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False, focal=False, CE=False,dice=False, channels=1):
    """Evaluation without the densecrf with the dice coefficient"""
    tot = 0
    for i, (imgs, true_masks) in enumerate(dataset):
        imgs = Variable(imgs)
        true_masks = Variable(true_masks)
        if gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

        if focal and CE:
            mask_pred = net(imgs)[0]
            ch1 = mask_pred[0]
            ch2 = mask_pred[1]
            mask_pred = ch2 - ch1
            mask_pred = (mask_pred > 0.).float().gpu
            mask_pred = mask_pred.unsqueeze(0)
            # # only for 0.4
            # height = mask_pred.size(2)
            # width = mask_pred.size(3)
            # mask_pred = mask_pred.contiguous().view(mask_pred.size(0), mask_pred.size(1), -1)
            # mask_pred = mask_pred.transpose(1, 2)
            # mask_pred = mask_pred.contiguous().view(-1, mask_pred.size(2)).squeeze()
            # mask_pred = torch.argmax(F.log_softmax(mask_pred, dim=1), dim=1)
            # mask_pred = mask_pred.reshape(height, width).float()
        elif channels == 2:
            pass
        else:
            mask_pred = net(imgs)[0]
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

        # print(mask_pred.size())
        # print(true_masks.size())
        tot += dice_coeff(mask_pred, true_masks)[0]
    return tot / i

def calcul_iou_for_focal(net, dataset, gpu=False):
    tot = 0
    for i, (img, true_mask) in enumerate(dataset):
        img = Variable(img)
        if gpu:
            img = img.cuda()

        pred_mask = net(img).cpu()
        pred_mask = pred_mask.data.numpy()
        true_mask = true_mask.numpy()
        pred_mask = np.argmax(pred_mask, axis=1)
        pred_mask = pred_mask.reshape(-1)
        true_mask = true_mask.reshape(-1)
        tot += jsc(pred_mask, true_mask)

    return tot / i

