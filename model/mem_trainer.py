"""training container

author: Haixin wang
e-mail: haixinwa@gmail.com
"""

import torch
import torch.nn as nn
import os
import torchvision as tv


class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, normals, query, target):
        b, t, c, h, w = normals.shape
        normals = normals.view(-1, c, h, w)
        m_keys, m_values = self.model([normals], mode='m')
        _, c, h_, w_ = m_keys.shape
        _, cv, h_, w_ = m_values.shape
        m_keys = m_keys.view(b, c, t, h_, w_)
        m_values = m_values.view(b, cv, t, h_, w_)
        preds = self.model([query, m_keys, m_values], mode='c')
        multi_loss = self.loss(preds, target)
        return multi_loss, preds


class MEM_Trainer():
    def __init__(self, net, loss, loss_name, optimizer, ngpu, debug=True):
        self.net = net
        self.loss = loss
        self.loss_name = loss_name
        self.loss_value = None
        self.optimizer = optimizer
        self.debug = debug
        self.network = torch.nn.DataParallel(Network(self.net, self.loss), device_ids=list(range(ngpu)))
        self.network.train()
        self.network.cuda()
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        torch.backends.cudnn.benchmark = True

    def save_params(self, save_path):
        print("saving model to {}".format(save_path))
        with open(save_path, "wb") as f:
            params = self.net.state_dict()
            torch.save(params, f)

    def load_params(self, path):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        w_dict = torch.load(path)
        for k, v in w_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)

    def set_lr(self, lr):
        # print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self, normals, query, target):
        normals = normals.cuda()
        query = query.cuda()
        target = target.cuda()
        if self.loss_name == 'SSIM_loss' or self.loss_name == 'VAE_loss':
            self.optimizer.zero_grad()
            loss, preds = self.network(normals, query, target)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.loss_value = loss.item()

        elif self.loss_name == 'Multi_SSIM_loss':
            self.loss_value = list()
            total_loss = list()
            self.optimizer.zero_grad()
            loss_multi, preds = self.network(normals, query, target)
            for loss in loss_multi:
                loss = loss.mean()
                total_loss.append(loss)
                self.loss_value.append(loss.item())
            total_loss = torch.stack(total_loss, 0).sum()
            total_loss.backward()
            self.optimizer.step()

        else:
            raise Exception('Wrong loss name')

        if self.debug:
            save_dir = './debug/SSIM'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            imgs = torch.cat((query, preds), 0)
            tv.utils.save_image(imgs, os.path.join(save_dir, 'ori_pred.jpg'),
                                normalize=True,
                                range=(-1, 1))

    def get_loss_message(self):
        if self.loss_name == 'SSIM_loss' or self.loss_name == 'VAE_loss':
            mes = 'ssim loss:{:.4f};'.format(self.loss_value)

        elif self.loss_name == 'Multi_SSIM_loss':
            mes = ''
            for k, loss in enumerate(self.loss_value):
                mes += 'size{:d} ssim loss:{:.4f}; '.format(k, loss)
        else:
            raise Exception('Wrong loss name')

        return mes
