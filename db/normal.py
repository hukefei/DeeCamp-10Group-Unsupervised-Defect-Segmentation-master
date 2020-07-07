import os
import re
import torch
import numpy as np
import random
import torch.utils.data as data
from collections import OrderedDict
from .augment import *
from .eval_func import *
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa

class Normal(data.dataset):
    def __init__(self, root, set='train', clip=3, target_size=(128, 128)):
        self.root = root
        self.set = set
        self.clip = clip
        self.target_size = target_size
        self.image_dir = os.path.join(root, set)

        self.images = os.listdir(self.image_dir)

    def __getitem__(self, item):
        idx_list = []
        idx_list.append(item)
        for _ in range(self.clip - 1):
            idx = random.choice(range(len(self.images)))
            idx_list.append(idx)
        img_list = []
        for idx in idx_list:
            img_f = self.images[idx]
            img_ = Image.open(os.path.join(self.image_dir, img_f)).resize(self.target_size)
            img_a = np.array(img_, dtype=np.uint8)
            img_t = torch.from_numpy(img_a)
            if idx == item:
                target = img_t
            img_t = img_t.unsqueeze(0)
            img_list.append(img_t)

        imgs = torch.cat(img_list, dim=0)

        return imgs, target

    def __len__(self):
        return len(self.images)

    def aug(self, image, seed):
        ia.seed(seed)

        # Example batch of images.
        # The array has shape (32, 64, 64, 3) and dtype uint8.
        images = image  # B,H,W,C

        # print('In Aug',images.shape,masks.shape)
        combo = np.concatenate((images, masks), axis=3)
        # print('COMBO: ',combo.shape)

        seq_all = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=False)  # apply augmenters in random order

        seq_f = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.01))
                          ),
            # iaa.contrast.LinearContrast((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
        ], random_order=False)

        combo_aug = seq_all(images=combo)
        # print('combo_au: ',combo_aug.shape)
        images_aug = combo_aug[:, :, :, :3]
        masks_aug = combo_aug[:, :, :, 3:]
        images_aug = seq_f(images=images_aug)

        return images_aug, masks_aug