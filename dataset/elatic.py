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

class ElaticSet(data.Dataset):
    """docstring for ElaticSet"""
    def __init__(self, root, transforms=None, train=True, test=False, deploy=False):
        super(ElaticSet, self).__init__()
        self.test = test
        self.deploy = deploy
        self.root = root

        if test:
            self.imgs = glob.glob(root + '/test' + "/*." + 'bmp')
        elif train:
            self.imgs = glob.glob(root + '/train' + "/*." + 'bmp')
        else:
            self.imgs = glob.glob(root + '/deploy' + "/*." + 'bmp')

        if transforms is None:
            normalize = T.Normalize(mean=[0.24], std=[0.11])
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
        img_path = self.imgs[index]
        label_path = ''
        
        if self.deploy: 
            label = None
        else:
            filename = img_path.split('/')[-1]
            label_path = self.root + '/label/' + filename
            
        data = cv2.imread(img_path, 0)
        data = data[:,:, np.newaxis]
        label = cv2.imread(label_path, 0)
        _, label = cv2.threshold(label, 100, 255, cv2.THRESH_BINARY)
        label = label[:, :, np.newaxis]
        # label_orig = cv2.threshold(label, 100, 255, cv2.THRESH_BINARY)
        # label_rev = cv2.threshold(label, 100, 255, cv2.THRESH_BINARY_INV)
        # label = np.concatenate((label_orig[:,:,np.newaxis], label_rev[:,:,np.newaxis]), axis=2)

        # 转换为灰度图像
        data = self.transforms4imgs(data)
        label = self.transforms4label(label)
        return data, label

    def __len__(self):
        return len(self.imgs)

    def getLen(self):
        return len(self.imgs)

