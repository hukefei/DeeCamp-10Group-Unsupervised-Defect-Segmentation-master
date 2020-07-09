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

class Memory(data.Dataset):
    def __init__(self, root, set='train', clip=5, transforms=None, transforms_query=None):
        super(Memory, self).__init__()
        self.root = root
        self.set = set
        self.clip = clip
        self.transforms = transforms
        self.transforms_query = transforms_query
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
            img_ = Image.open(os.path.join(self.image_dir, img_f))
            if idx == item:
                target = img_
                query = img_
            img_n = self.transforms(img_)
            img_list.append(img_n)
        query = self.transforms_query(query)
        target = self.transforms_query(target)
        img_list = [img.unsqueeze(0) for img in img_list]
        imgs = torch.cat(img_list, dim=0)
        sample = {'normal': imgs, 'query': query, 'target': target}

        return sample

    def __len__(self):
        return len(self.images)
