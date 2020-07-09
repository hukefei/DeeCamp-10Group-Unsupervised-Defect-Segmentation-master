from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

# draw
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print('Memory Networks: initialized.')


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        # self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485]).view(1, 1, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229]).view(1, 1, 1, 1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std
        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f, x


class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.456]).view(1, 1, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229]).view(1, 1, 1, 1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std
        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f, x


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, indim, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(indim, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.tanh = nn.Tanh()

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        p3 = self.pred2(F.relu(m3))
        p4 = self.pred2(F.relu(m4))

        p2 = self.tanh(p2)
        p3 = self.tanh(p3)
        p4 = self.tanh(p4)

        p2 = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, scale_factor=16, mode='bilinear', align_corners=False)
        return p2# , p2, p3, p4


class BNConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BNConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BNDeConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, out_padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False, scale_factor=2):
        super(BNDeConv, self).__init__()
        self.out_channels = out_planes
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        # self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                                output_padding=out_padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Decoder_2(nn.Module):
    def __init__(self, code_dim, img_channel):
        super(Decoder_2, self).__init__()
        self.deconv1 = BNConv(in_planes=code_dim, out_planes=128, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv2 = BNDeConv(in_planes=128, out_planes=128, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv3 = BNDeConv(in_planes=128, out_planes=128, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation3 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv4 = BNConv(in_planes=128, out_planes=128, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation4 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv5 = BNDeConv(in_planes=128, out_planes=64, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation5 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv6 = BNConv(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation6 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv7 = BNDeConv(in_planes=64, out_planes=32, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation7 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv8 = BNConv(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation8 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        # self.deconv9 = BNDeConv(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1, relu=False)
        # self.activation9 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv10 = BNConv(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation10 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv11 = BNConv(in_planes=32, out_planes=img_channel, kernel_size=1, stride=1, relu=False)
        self.activation11 = nn.Sigmoid()

    def forward(self, x):
        x = self.activation1(self.deconv1(x))
        x = self.activation2(self.deconv2(x))
        x = self.activation3(self.deconv3(x))
        x = self.activation4(self.deconv4(x))
        x = self.activation5(self.deconv5(x))
        x = self.activation6(self.deconv6(x))
        x = self.activation7(self.deconv7(x))
        x = self.activation8(self.deconv8(x))
        # x = self.activation9(self.deconv9(x))
        x = self.activation10(self.deconv10(x))
        x = self.activation11(self.deconv11(x))

        return x


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, keys_m, values_m, key_q, value_q):
        '''
        :param keys_m: [B,C,T,H,W], c = 128
        :param values_m: [B,C,T,H,W], c = 512
        :param key_q: [B,C,H,W], c = 128
        :param value_q: [B,C,H,W], c = 512
        :return: final_value [B, C, H, W]
        '''
        B, C_key, T, H, W = keys_m.size()
        # print('#####', B, C_key, T, H, W)
        _, C_value, _, _, _ = values_m.size()

        keys_m_temp = keys_m.view(B, C_key, T * H * W)
        keys_m_temp = torch.transpose(keys_m_temp, 1, 2)  # [b,thw,c]

        key_q_temp = key_q.view(B, C_key, H * W)  # [b,c,hw]

        p = torch.bmm(keys_m_temp, key_q_temp)  # [b, thw, hw]
        p = p / math.sqrt(C_key)
        p = F.softmax(p, dim=1)  # b, thw, hw

        mo = values_m.view(B, C_value, T * H * W)  # [b,c,thw]
        mem = torch.bmm(mo, p)  # Weighted-sum B, c, hw
        mem = mem.view(B, C_value, H, W)

        final_value = torch.cat([mem, value_q], dim=1)
        # print('mem:', torch.max(mem), torch.min(mem))
        # print('value_q:', torch.max(value_q), torch.min(value_q))

        return final_value


class STM(nn.Module):
    def __init__(self):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M()
        self.Encoder_Q = Encoder_Q()

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(1024, 256)

    def contrust(self, frame, key, value):
        '''
        recontrust frame
        :param frame:
        :param key:
        :param value:
        :return:
        '''
        # encode
        r4, r3, r2, _, _, x = self.Encoder_Q(frame)
        curKey, curValue = self.KV_Q_r4(r4)  # 1, dim, H/16, W/16

        # memory select
        final_value = self.Memory(key, value, curKey, curValue)
        p_m2 = self.Decoder(final_value, r3, r2)
        return p_m2

    def segment(self, frame, key, value):
        '''
        :param frame: 当前需要分割的image；[B,C,H,W]
        :param key: 当前memory的key；[B,C,T,H,W]
        :param value: 当前memory的value; [B,C,T,H,W]
        :return: logits []
        '''
        # encode
        r4, r3, r2, _, _, x = self.Encoder_Q(frame)
        curKey, curValue = self.KV_Q_r4(r4)  # 1, dim, H/16, W/16

        # memory select
        final_value = self.Memory(key, value, curKey, curValue)
        logits = self.Decoder(final_value, r3, r2)  # [b,2,h,w]
        return logits

    def memorize(self, curFrame):
        '''
        将当前帧编码
        :param curFrame: [b,c,h,w]
        :param curMask: [b,c,h,w]
        :return: 编码后的key与value
        '''
        # print('&&&&&&&&&', curMask.shape, curFrame.shape)
        r4, _, _, _, _, x = self.Encoder_M(curFrame)
        # r4 = r4.detach().cpu().numpy()
        # print('r4.shape', r4.shape)
        #
        # file_name = './exp/pretrained/tf/encoder_m/dogs_jump_1_frame1/'
        #
        #
        # if not os.path.exists(file_name):
        #     os.makedirs(file_name)
        # np.save('./exp/pretrained/tf/encoder_m/dogs_jump_1_frame1.npy', r4)
        # sns.set()
        # for c in range(len(r4[0])):
        #     if c>300:
        #         break
        #     plt.clf()
        #     ax = sns.heatmap(r4[0, c])
        #     plt.savefig(file_name+'{:05d}.png'.format(c))
        # sys.exit()

        # print('******r4', r4.device)
        k4, v4 = self.KV_M_r4(r4)  # num_objects, 128 and 512, H/16, W/16
        return k4, v4

    def forward(self, args, mode='m'):
        # args: Fs[:,:,t-1]
        # kwargs: Es[:,:,t-1]
        assert mode in ('m', 'c')
        if mode == 'c':
            return self.contrust(args[0], args[1], args[2])
        else:
            return self.memorize(args[0])
