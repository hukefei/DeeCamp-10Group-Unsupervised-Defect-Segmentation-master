import torch
import torch.nn as nn
import torch.nn.functional as F

class Perceptual_loss(nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        # conv5_1 = nn.Conv2d(512, 512, 3, stride=1, pad=1),
        # conv5_2 = nn.Conv2d(512, 512, 3, stride=1, pad=1),
        # conv5_3 = nn.Conv2d(512, 512, 3, stride=1, pad=1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.mse_loss_2 = nn.MSELoss(reduction='none')

    def get_feat(self, x):
        y1 = self.relu(self.conv1_2(self.relu(self.conv1_1(x))))
        h = self.maxpool2d(y1)
        y2 = self.relu(self.conv2_2(self.relu(self.conv2_1(h))))
        h = self.maxpool2d(y2)
        y3 = self.relu(self.conv3_3(self.relu(self.conv3_2(self.relu(self.conv3_1(h))))))
        h = self.maxpool2d(y3)
        y4 = self.relu(self.conv4_3(self.relu(self.conv4_2(self.relu(self.conv4_1(h))))))
        return [y1, y2, y3, y4]

    def forward(self, input1, input2):
        f1 = self.get_feat(input1)
        f2 = self.get_feat(input2)

        loss = self.mse_loss(input1, input2) + self.mse_loss(f1[0], f2[0]) + 0.5 * self.mse_loss(f1[1], f2[1]) + 0.25 * self.mse_loss(f1[2], f2[2])

        return loss

    def res(self, input1, input2):
        f1 = self.get_feat(input1)
        f2 = self.get_feat(input2)

        res0 = self.mse_loss_2(input1, input2).mean(1).unsqueeze(0)
        res1 = self.mse_loss_2(f1[0], f2[0]).mean(1).unsqueeze(0)
        res2 = self.mse_loss_2(f1[1], f2[1]).mean(1).unsqueeze(0)
        res3 = self.mse_loss_2(f1[2], f2[2]).mean(1).unsqueeze(0)

        res = res0 + res1 + F.interpolate(res2, scale_factor=2, mode='bilinear', align_corners=False) \
            + F.interpolate(res3, scale_factor=4, mode='bilinear', align_corners=False)

        res = res.squeeze(0).squeeze(0).detach().cpu().numpy()

        return res