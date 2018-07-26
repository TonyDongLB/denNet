import cv2
import os
from PIL import Image
from torch.utils import data
import numpy as np
import torch
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
import glob
import cv2

class Hand(data.Dataset):
    ''' to load the hand dataset'''
    def __init__(self, root, transforms=None, train=True, test=False, deploy=False):
        super(Hand, self).__init__()
        self.test = test
        self.deploy = deploy
        self.root = root

        if test:
            self.imgs = glob.glob(root + '/test' + "/*." + 'jpg')
        elif train:
            self.imgs = glob.glob(root + '/train' + "/*." + 'jpg')
        else:
            self.imgs = glob.glob(root + '/deploy' + "/*." + 'jpg')

        self.imgs = [(path, i) for i in range(2) for path in self.imgs]

        if transforms is None:
            # normalize need to edit!
            normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.4, 0.4, 0.4])
            self.transforms4imgs = T.Compose([
                T.ToTensor(),
                normalize
            ])
            self.transforms4label = T.Compose([
                T.ToTensor(),
            ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.imgs[index][0]
        side = self.imgs[index][1]
        label_path = ''

        if self.deploy:
            label = None
        else:
            filename = img_path.split('/')[-1]
            pre = filename.split('.')[0]
            suffix = filename.split('.')[-1]

            label_path = self.root + '/label/' + pre + '_mask.' + suffix

        data = cv2.imread(img_path)
        label = cv2.imread(label_path, 0)

        height = data.shape[0]
        width = data.shape[1]
        if width > height:
            if side == 0:
                data = data[:, :height]
                label = label[:, :height]
            else:
                data = data[:, width - height:]
                label = label[:, width - height:]
        if height > width:
            if side == 0:
                data = data[:width, :]
                label = label[:width, :]
            else:
                data = data[height - width:]
                label = label[height - width:]

        _, label = cv2.threshold(label, 100, 255, cv2.THRESH_BINARY)
        label = label / 255
        label = label[:, :, np.newaxis]

        data = self.transforms4imgs(data)
        label = self.transforms4label(label)
        return data, label

    def __len__(self):
        return len(self.imgs)

    def getLen(self):
        return len(self.imgs)


