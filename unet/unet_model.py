# full assembly of the sub-parts to form the complete net

from .unet_parts import *
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, reduce=False, SE_mode=False):
        ''' the reduce means reduce the second feature map in up function, because
            we think classification is more important '''
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128, SE_mode=SE_mode)
        self.down2 = down(128, 256, SE_mode=SE_mode)
        self.down3 = down(256, 512, SE_mode=SE_mode)
        self.down4 = down(512, 1024, SE_mode=SE_mode)
        if reduce:
            self.up1 = re_up(1024, 512, SE_mode=SE_mode)
            self.up2 = re_up(512, 256, SE_mode=SE_mode)
            self.up3 = re_up(256, 128, SE_mode=SE_mode)
            self.up4 = re_up(128, 64, SE_mode=SE_mode)
            self.outc = outconv(64, n_classes)
        else:
            self.up1 = up(1024, 512, SE_mode=SE_mode)
            self.up2 = up(512, 256, SE_mode=SE_mode)
            self.up3 = up(256, 128, SE_mode=SE_mode)
            self.up4 = up(128, 64, SE_mode=SE_mode)
            self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# # it seems that it perform not that good as a more deeper model
class DeeperUNet(nn.Module):
    def __init__(self, n_channels, n_classes, reduce=False, SE_mode=False):
        super(DeeperUNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128, SE_mode=SE_mode)
        self.down2 = down(128, 256, SE_mode=SE_mode)
        self.down3 = down(256, 512, SE_mode=SE_mode)
        self.down4 = down(512, 1024, SE_mode=SE_mode)
        self.down5 = down(1024, 2048, SE_mode=SE_mode)
        if reduce:
            self.up1 = re_up(1024, 512, SE_mode=SE_mode)
            self.up2 = re_up(512, 256, SE_mode=SE_mode)
            self.up3 = re_up(256, 128, SE_mode=SE_mode)
            self.up4 = re_up(128, 64, SE_mode=SE_mode)
            self.outc = outconv(64, n_classes)
        else:
            self.up1 = up(2048, 1024,SE_mode=SE_mode)
            self.up2 = up(1024, 512, SE_mode=SE_mode)
            self.up3 = up(512, 256, SE_mode=SE_mode)
            self.up4 = up(256, 128, SE_mode=SE_mode)
            self.up5 = up(128, 64, SE_mode=SE_mode)
            self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        return x
