import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from eval import eval_net
from unet import UNet

from dataset import Hand
import unet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from PIL import Image
from torchvision import transforms as T
from focal_loss import FocalLoss

from focalloss2d import FocalLoss2d


def train_net(net,
              epochs=20,
              batch_size=32,
              lr=0.1,
              save_cp=True,
              gpu=True,
              img_scale=0.5):
    root_data = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data/hand/')
    dir_checkpoint = 'checkpoints/'

    train_set = Hand(root_data, train=True)
    test_set = Hand(root_data, test=True)
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=12)
    test_data = DataLoader(test_set, batch_size=1, shuffle=False,
                           num_workers=12)

    N_train = train_set.getLen()
    N_test = test_set.getLen()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr,
        weight_decay=1e-3)

    # optimizer = optim.SGD(net.parameters(),
    #                       lr=lr,
    #                       momentum=0.9,
    #                       weight_decay=0.0005)

    # to use focal loss
    # criterion = FocalLoss(class_num=2,gamma=1)

    # to use CEloss with weight
    # weight = torch.Tensor([1, 99])
    # if gpu:
    #     weight = weight.cuda()
    # criterion = torch.nn.CrossEntropyLoss(weight=weight)

    # to use BCE loss
    criterion = nn.BCELoss()

    writer = SummaryWriter('log/hand_batch8')
    processed_batch = 0

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, N_train,
               N_test, str(save_cp), str(gpu)))

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        epoch_loss = 0
        num_i = 0
        for ii, (imgs, true_masks) in enumerate(train_data):
            num_i += 1
            processed_batch += 1

            imgs = Variable(imgs)
            true_masks = Variable(true_masks)
            if gpu:
                # input = input.long()

                # true_masks = true_masks.long()
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            # to use BCE loss
            optimizer.zero_grad()
            masks_pred = net(imgs)
            masks_pred = F.sigmoid(masks_pred)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)

            # to use classification loss
            # masks_pred = masks_pred.contiguous().view(masks_pred.size(0), masks_pred.size(1), -1)
            # masks_pred = masks_pred.transpose(1, 2)
            # masks_pred = masks_pred.contiguous().view(-1, masks_pred.size(2)).squeeze()
            # true_masks = true_masks.contiguous().view(true_masks.size(0), true_masks.size(1), -1)
            # true_masks = true_masks.transpose(1, 2)
            # true_masks = true_masks.contiguous().view(-1, true_masks.size(2)).squeeze()
            # loss = criterion(masks_pred, true_masks)

            epoch_loss += loss.data[0]

            print('{0:.4f} --- loss: {1:.6f}'.format(ii * batch_size / N_train, loss.data[0]))
            writer.add_scalar('loss', loss.data[0], processed_batch)

            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / num_i))
        writer.add_scalar('train_loss_epoch', epoch_loss / num_i, epoch + 1)

        val_dice = eval_net(net, test_data, gpu)
        print('Validation Dice Coeff: {}'.format(val_dice))
        writer.add_scalar('val_dice', val_dice, epoch + 1)

        if save_cp and val_dice > 0.95:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}_hand_new.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=500, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    import os;

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu, )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
