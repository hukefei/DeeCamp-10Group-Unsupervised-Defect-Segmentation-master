"""Data set tool of MVTEC

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import os
import re
import cv2
import random
import torch
import torch.utils.data as data
from collections import OrderedDict
from .augment import *
from .eval_func import *


class Preproc(object):
    """Pre-procession of input image includes resize, crop & data augmentation

    Arguments:
        resize: tup(int width, int height): resize shape
        crop: tup(int width, int height): crop shape
    """
    def __init__(self, resize):
        self.resize = resize

    def __call__(self, image):
        image = cv2.resize(image, self.resize)
        # random transformation
        p = random.uniform(0, 1)
        if (p > 0.2) and (p <= 0.4):
            image = mirror(image)
        elif (p > 0.4) and (p <= 0.6):
            image = flip(image)
        # elif (p > 0.6) and (p <= 0.8):
        #     image = shift(image, (-12, 12))
        # else:
        #     image = rotation(image, (-10, 10))

        # light adjustment
        p = random.uniform(0, 1)
        if p > 0.5:
            image = lighting_adjust(image, k=(0.8, 0.95), b=(-10, 10))

        # image normal
        image = image.astype(np.float32) / 255.
        # normalize_(tile, self.mean, self.std)
        image = torch.from_numpy(image)

        return image.unsqueeze(0)


class CHIP(data.Dataset):
    """A tiny data set for chip cell

    Arguments:
        root (string): root directory to root folder.
        set (string): image set to use ('train', or 'test')
        preproc(callable, optional): pre-procession on the input image
    """

    def __init__(self, root, set, preproc=None):
        self.root = root
        self.preproc = preproc
        self.set = set

        if set == 'train':
            self.ids = list()
            set_path = os.path.join(self.root, set)
            for img in os.listdir(set_path):
                item_path = os.path.join(set_path, img)
                self.ids.append(item_path)
        elif set == 'test':
            self.test_len = 0
            self.test_dict = OrderedDict()
            set_path = os.path.join(self.root, set)
            for type in os.listdir(set_path):
                type_dir = os.path.join(set_path, type)
                if os.path.isfile(type_dir):
                    continue
                ids = list()
                for img in os.listdir(type_dir):
                    if re.search('.png', img) is None:
                        continue
                    ids.append(os.path.join(type_dir, img))
                    self.test_len += 1
                self.test_dict[type] = ids
        else:
            raise Exception("Invalid set name")

    def __getitem__(self, index):
        """Returns training image
        """
        img_path = self.ids[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = self.preproc(img)

        return img

    def __len__(self):
        if self.set == 'train':
            return len(self.ids)
        else:
            return self.test_len

    def eval(self, eval_dir):
        summary_file = open(os.path.join(eval_dir, 'summary.txt'), 'w')
        for item in self.test_dict:
            if item != ".ipynb_checkpoints":
                summary_file.write('--------------{}--------------\n'.format(item))
                s_map_all = torch.load(eval_dir + '/' + '/' + 's_map.pth')
                labels = list()
                paccs = list()
                ious = list()
                FPR_list = list()
                TPR_list = list()
                gt_re_list = list()
                good_num = 0
                num_total = 0
                gt_dir = os.path.join(self.root, 'ground_truth')
                res_dir = os.path.join(eval_dir, item, 'mask')
                log_file = open(os.path.join(eval_dir, item, 'result.txt'), 'w')
                log_file.write('Item: {}\n'.format(item))
                type_ious = list()
                type_paccs = list()
                image_count = 0
                for mask_type in os.listdir(res_dir):
                    log_file.write('--------------------------\nType: {}\n'.format(mask_type))
                    image_count += 1
                    if item != 'good':
                        mask_id = mask_type.split('.')[0]
                        gt_id = mask_type.split('.')[0]  # os.listdir(gt_dir)[image_count].split('.')[0]
                        gt = cv2.imread(os.path.join(gt_dir, '{}.png').format(gt_id))
                        mask = cv2.imread(os.path.join(res_dir, '{}.png').format(mask_id))
                        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        _, gt = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
                        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                        _h, _w = gt.shape
                        mask = cv2.resize(mask, (_w, _h))
                        labels.append(0)
                        type_ious.append(cal_iou(mask, gt))
                        gt_re_list.append(gt.reshape(_w * _h, 1))
                    elif item == 'good':
                        num_total += 1
                        mask_id = mask_type.split('.')[0]
                        mask = cv2.imread(os.path.join(res_dir, '{}.png').format(mask_id))
                        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                        pixel_defect = (mask == 255)
                        num_pixel = float(pixel_defect.sum())
                        if num_pixel <= 10485.76:
                            good_num += 1
                        continue
                    else:
                        raise Exception ("invalid item name")
                    type_paccs.append(cal_pixel_accuracy(mask, gt))

                if item == 'good':
                    acc_good = good_num / num_total
                    log_file.write('mean IoU: nan\n')
                    log_file.write('classification accuracy of good samples:{:2f}\n'.format(acc_good* 100))
                elif item == 'bad':
                    log_file.write('mean IoU:{:.2f}\n'.format(np.array(type_ious).mean() * 100))
                    log_file.write('mean Pixel Accuracy:{:2f}\n'.format(np.array(type_paccs).mean() * 100))
                    ious += type_ious
                    paccs += type_paccs
                    mIoU = np.array(ious).mean()
                    # mPAc = np.array(paccs).mean()

                    s_map_all = np.array(s_map_all).reshape(-1, 1)
                    gt_re = np.array(gt_re_list)
                    gt_re = gt_re.reshape(-1, 1)
                    for threshold in np.arange(0, 1, 0.005):
                        FPR_list.append(cal_FPR(s_map_all, gt_re, threshold))
                        TPR_list.append(cal_TPR(s_map_all, gt_re, threshold))

                    auc = cal_AUC(TPR_list, FPR_list)
                    plt.figure()
                    plt.plot(FPR_list, TPR_list, '.-')
                    plt.savefig('./eval_result/' + item + '/ROC_curve/' + 'roc.jpg')
                    torch.save(type_ious, os.path.join(eval_dir) + '/type_ious.pth')
                    log_file.write('--------------------------\n')
                    log_file.write('Total mean IoU:{:.2f}\n'.format(mIoU * 100))
                    log_file.write('AUC of segmentation: {:.2f}\n'.format(auc * 100))
                    summary_file.write('mIoU:{:.2f}  auc:{:.2f}\n'.format(mIoU * 100, auc * 100))
                    log_file.write('\n')
                    log_file.close()
                    pass
                else:
                    raise Exception("invalid item name")
            elif item == ".ipynb_checkpoints":
                pass
            else:
                raise Exception("invalid folder name")