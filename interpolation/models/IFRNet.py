import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import warp, get_robust_weight
from loss import *

def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias = bias),
                         nn.PReLU(out_channels))


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                                   nn.PReLU(in_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                                   nn.PReLU(in_channels))
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.prelu(x + self.conv3(out))
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(convrelu(1, 32, 3, 2, 1),
                                      convrelu(32, 32, 3, 1, 1))
        self.pyramid2 = nn.Sequential(convrelu(32, 48, 3, 2, 1),
                                      convrelu(48, 48, 3, 1, 1))
        self.pyramid3 = nn.Sequential(convrelu(48, 72, 3, 2, 1),
                                      convrelu(72, 72, 3, 1, 1))
        self.pyramid4 = nn.Sequential(convrelu(72, 96, 3, 2, 1),
                                      convrelu(96, 96, 3, 1, 1))

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4

class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(convrelu(192+1, 192),
                                       ResBlock(192, 32),
                                       nn.ConvTranspose2d(192, 76, 4, 2, 1, bias=True))

    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        f_out = self.convblock(f_in)
        return f_out

class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(convrelu(220, 216),
                                       ResBlock(216, 32),
                                       nn.ConvTranspose2d(216, 52, 4, 2, 1, bias=True))

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out

class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(convrelu(148, 144),
                                       ResBlock(144, 32),
                                       nn.ConvTranspose2d(144, 36, 4, 2, 1, bias=True))
    
    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out

class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(convrelu(100, 96),
                                       ResBlock(96, 32),
                                       nn.ConvTranspose2d(96, 6, 4, 2, 1, bias=True))
    
    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7, device)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3, device)

    def inference(self, img0, img1, embt):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        _, _, h, w = img0.shape

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)

        out4 = self.decoder4(f0_4, f1_4, embt)
        _, _, h4, w4 = out4.shape
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        _, _, h3, w3 = out3.shape
        up_flow0_3 = out3[:, 0:2] + (w3 / w4) * resize(up_flow0_4, size = (h3, w3))
        up_flow1_3 = out3[:, 2:4] + (w3 / w4) * resize(up_flow1_4, size = (h3, w3))
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        _, _, h2, w2 = out2.shape
        up_flow0_2 = out2[:, 0:2] + (w2 / w3) * resize(up_flow0_3, size = (h2, w2))
        up_flow1_2 = out2[:, 2:4] + (w2 / w3) * resize(up_flow1_3, size = (h2, w2))
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        _, _, h1, w1 = out1.shape
        up_flow0_1 = out1[:, 0:2] + (w1 / w2) * resize(up_flow0_2, size = (h1, w1))
        up_flow1_1 = out1[:, 2:4] + (w1 / w2) * resize(up_flow1_2, size = (h1, w1))
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]

        up_flow0_1 = resize(up_flow0_1, size = (h, w)) * (w / w1)
        up_flow1_1 = resize(up_flow1_1, size = (h, w)) * (w / w1)
        up_mask_1 = resize(up_mask_1, size = (h, w))
        up_res_1 = resize(up_res_1, size = (h, w))

        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1)*img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        return imgt_pred

    def forward(self, img0, img1, embt, imgt):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        imgt_ = imgt - mean_
        _, _, h, w = img0.shape

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        ft_1, ft_2, ft_3, ft_4 = self.encoder(imgt_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        _, _, h4, w4 = out4.shape
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]
        
        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        _, _, h3, w3 = out3.shape
        up_flow0_3 = out3[:, 0:2] + (w3 / w4) * resize(up_flow0_4, size = (h3, w3))
        up_flow1_3 = out3[:, 2:4] + (w3 / w4) * resize(up_flow1_4, size = (h3, w3))
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        _, _, h2, w2 = out2.shape
        up_flow0_2 = out2[:, 0:2] + (w2 / w3) * resize(up_flow0_3, size = (h2, w2))
        up_flow1_2 = out2[:, 2:4] + (w2 / w3) * resize(up_flow1_3, size = (h2, w2))
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        _, _, h1, w1 = out1.shape
        up_flow0_1 = out1[:, 0:2] + (w1 / w2) * resize(up_flow0_2, size = (h1, w1))
        up_flow1_1 = out1[:, 2:4] + (w1 / w2) * resize(up_flow1_2, size = (h1, w1))
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        
        up_flow0_1 = resize(up_flow0_1, size = (h, w)) * (w / w1)
        up_flow1_1 = resize(up_flow1_1, size = (h, w)) * (w / w1)
        up_mask_1 = resize(up_mask_1, size = (h, w))
        up_res_1 = resize(up_res_1, size = (h, w))

        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1)*img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1

        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        ft_1_ = resize(ft_1_, size = (ft_1.shape[-2], ft_1.shape[-1])) * (ft_1.shape[-1] / w2)
        ft_2_ = resize(ft_2_, size = (ft_2.shape[-2], ft_2.shape[-1])) * (ft_2.shape[-1] / w3)
        ft_3_ = resize(ft_3_, size = (ft_3.shape[-2], ft_3.shape[-1])) * (ft_3.shape[-1] / w4)
        loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))

        return imgt_pred, loss_rec, loss_geo