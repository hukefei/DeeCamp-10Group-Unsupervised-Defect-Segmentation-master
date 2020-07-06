import os
import cv2
import json
import argparse
import numpy as np
from db import Transform, Transform_chip
from model.rebuilder import Rebuilder
from model.segmentation import ssim_seg, ssim_seg_mvtec, seg_mask, seg_mask_mvtec
from tools import Timer
from factory import *
from db.eval_func import cal_good_index


def parse_args():
    parser = argparse.ArgumentParser(description='Object detection base on anchor.')
    parser.add_argument('--cfg', help="Path of config file", type=str, required=True)
    parser.add_argument('--model_path', help="Path of model", type=str,required=True)
    parser.add_argument('--gpu_id', help="ID of GPU", type=int, default=0)
    parser.add_argument('--res_dir', help="Directory path of result", type=str, default='./eval_result')
    parser.add_argument('--retest', default=False, type=bool)

    return parser.parse_args()

def val_mvtec(rebuilder, transform, configs):
    if configs['db']['use_validation_set'] is True:
        val_set = load_data_set_from_factory(configs, 'validation')
        print('Data set: {} has been loaded'.format(configs['db']['name']))
        threshold_seg_dict = dict()
        for item in val_set.val_dict:
            item_list = val_set.val_dict[item]
            good_count = 0
            for threshold_temp in range(0, 256):
                for path in item_list:
                    image = cv2.imread(path, cv2.IMREAD_COLOR)
                    ori_h, ori_w, _ = image.shape
                    ori_img, input_tensor = transform(image)
                    out = rebuilder.inference(input_tensor)
                    re_img = out.transpose((1, 2, 0))
                    s_map = ssim_seg(ori_img, re_img)
                    s_map = cv2.resize(s_map, (ori_w, ori_h))
                    mask = seg_mask(s_map, threshold_temp)
                    good_count += cal_good_index(mask, 400)
                if good_count >= int(0.99 * len(item_list)):
                    threshold_seg_dict[item] = threshold_temp
                    break
            print('validation: Item:{} finishes'.format(item))
    elif configs['db']['use_validation_set'] is False:
        print("validation set is not used")
        threshold_seg_dict = dict()
    else:
        raise Exception("invalid input")
    return threshold_seg_dict


def test_mvtec(test_set, rebuilder, transform, save_dir, threshold_seg_dict, configs):
    _t = Timer()
    cost_time = list()
    if not os.path.exists(os.path.join(save_dir, 'ROC_curve')):
        os.mkdir(os.path.join(save_dir, 'ROC_curve'))
    for item in test_set.test_dict:
        s_map_list = list()
        s_map_good_list=list()
        item_dict = test_set.test_dict[item]

        if not os.path.exists(os.path.join(save_dir, item)):
            os.mkdir(os.path.join(save_dir, item))
            os.mkdir(os.path.join(save_dir, item, 'ori'))
            os.mkdir(os.path.join(save_dir, item, 'gen'))
            os.mkdir(os.path.join(save_dir, item, 'mask'))
            #os.mkdir(os.path.join(save_dir, item))
        for type in item_dict:
            if not os.path.exists(os.path.join(save_dir, item, 'ori', type)):
                os.mkdir(os.path.join(save_dir, item, 'ori', type))
            if not os.path.exists(os.path.join(save_dir, item, 'gen', type)):
                os.mkdir(os.path.join(save_dir, item, 'gen', type))
            if not os.path.exists(os.path.join(save_dir, item, 'mask', type)):
                os.mkdir(os.path.join(save_dir, item, 'mask', type))
            _time = list()
            img_list = item_dict[type]
            for path in img_list:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                ori_h, ori_w, _ = image.shape
                _t.tic()
                ori_img, input_tensor = transform(image)
                out = rebuilder.inference(input_tensor)
                re_img = out.transpose((1, 2, 0))
                s_map = ssim_seg_mvtec(ori_img, re_img, configs, win_size=3, gaussian_weights=True)
                if threshold_seg_dict: # dict is not empty
                    mask = seg_mask_mvtec(s_map, threshold_seg_dict[item],configs)
                else:
                    mask = seg_mask_mvtec(s_map, 64, configs)
                inference_time = _t.toc()
                img_id = path.split('.')[0][-3:]
                cv2.imwrite(os.path.join(save_dir, item, 'ori', type, '{}.png'.format(img_id)), ori_img)
                cv2.imwrite(os.path.join(save_dir, item, 'gen', type, '{}.png'.format(img_id)), re_img)
                cv2.imwrite(os.path.join(save_dir, item, 'mask', type, '{}.png'.format(img_id)), mask)
                _time.append(inference_time)
                if type != 'good':
                    s_map_bad=s_map.reshape(-1,1)
                    s_map_list.append(s_map_bad)
                else:
                    s_map_good = s_map.reshape(-1, 1)
                    s_map_good_list.append(s_map_good)
            cost_time += _time
            mean_time = np.array(_time).mean()
            print('Evaluate: Item:{}; Type:{}; Mean time:{:.1f}ms'.format(item, type, mean_time*1000))
            _t.clear()
        torch.save(s_map_list, os.path.join(save_dir, item) + '/s_map.pth')
        torch.save(s_map_good_list, os.path.join(save_dir, item) + '/s_map_good.pth')

    # calculate mean time
    cost_time = np.array(cost_time)
    cost_time = np.sort(cost_time)
    num = cost_time.shape[0]
    num90 = int(num*0.9)
    cost_time = cost_time[0:num90]
    mean_time = np.mean(cost_time)
    print('Mean_time: {:.1f}ms'.format(mean_time*1000))

    # evaluate results
    print('Evaluating...')
    test_set.eval(save_dir)


def test_chip(test_set, rebuilder, transform, save_dir, configs):
    _t = Timer()
    cost_time = list()
    iou_list={}
    s_map_list=list()
    for type in test_set.test_dict:
        img_list = test_set.test_dict[type]
        if not os.path.exists(os.path.join(save_dir, type)):
            os.mkdir(os.path.join(save_dir, type))
            os.mkdir(os.path.join(save_dir, type, 'ori'))
            os.mkdir(os.path.join(save_dir, type, 'gen'))
            os.mkdir(os.path.join(save_dir, type, 'mask'))
        if not os.path.exists(os.path.join(save_dir, type,'ROC_curve')):
            os.mkdir(os.path.join(save_dir, type, 'ROC_curve'))
        for k, path in enumerate(img_list):
            name= path.split('/')[-1]
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _t.tic()
            ori_img, input_tensor = transform(image)
            out = rebuilder.inference(input_tensor)
            re_img = out[0]
            s_map = ssim_seg(ori_img, re_img, win_size=11, gaussian_weights=True)
            _h, _w = image.shape
            s_map_save = cv2.resize(s_map, (_w, _h))
            s_map_list.append(s_map_save.reshape(-1,1))
            mask = seg_mask(s_map, threshold=128)
            inference_time = _t.toc()
            if configs['db']['resize'] == [832, 832]:
                #cat_img = np.concatenate((ori_img[32:-32,32:-32], re_img[32:-32,32:-32], mask[32:-32,32:-32]), axis=1)
                cv2.imwrite(os.path.join(save_dir, type, 'ori', 'mask{:d}.png'.format(k)), ori_img[32:-32,32:-32])
                cv2.imwrite(os.path.join(save_dir, type, 'gen', 'mask{:d}.png'.format(k)), re_img[32:-32,32:-32])
                cv2.imwrite(os.path.join(save_dir, type, 'mask', 'mask{:d}.png'.format(k)), mask[32:-32,32:-32])
            elif configs['db']['resize'] == [768, 768]:
                cv2.imwrite(os.path.join(save_dir, type, 'ori', 'mask{:d}.png'.format(k)), ori_img)
                cv2.imwrite(os.path.join(save_dir, type, 'gen', 'mask{:d}.png'.format(k)), re_img)
                cv2.imwrite(os.path.join(save_dir, type, 'mask', 'mask{:d}.png'.format(k)), mask)
            elif configs['db']['resize'] == [256, 256]:
                cv2.imwrite(os.path.join(save_dir, type, 'ori', name), ori_img)
                cv2.imwrite(os.path.join(save_dir, type, 'gen', name), re_img)
                cv2.imwrite(os.path.join(save_dir, type, 'mask', name), mask)
            else:
                raise Exception("invaild image size")
            #cv2.imwrite(os.path.join(save_dir, type, '{:d}.png'.format(k)), cat_img)
            cost_time.append(inference_time)
            if (k+1) % 20 == 0:
                print('{}th image, cost time: {:.1f}'.format(k+1, inference_time*1000))
            _t.clear()
        torch.save(s_map_list,os.path.join(save_dir) + '/s_map.pth')
    # calculate mean time
    cost_time = np.array(cost_time)
    cost_time = np.sort(cost_time)
    num = cost_time.shape[0]
    num90 = int(num*0.9)
    cost_time = cost_time[0:num90]
    mean_time = np.mean(cost_time)
    print('Mean_time: {:.1f}ms'.format(mean_time*1000))
    test_set.eval(save_dir)


if __name__ == '__main__':
    args = parse_args()

    # load config file
    cfg_file = os.path.join('./config', args.cfg + '.json')
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)


    # load data set
    test_set = load_data_set_from_factory(configs, 'test')
    print('Data set: {} has been loaded'.format(configs['db']['name']))

    # retest
    if args.retest is True:
        print('Evaluating...')
        test_set.eval(args.res_dir)
        exit(0)

    # init and load Rebuilder
    # load model
    if configs['db']['name'] == 'mvtec':
        transform = Transform(tuple(configs['db']['resize']))
    elif configs['db']['name'] == 'chip':
        transform = Transform_chip(tuple(configs['db']['resize']))
    else:
        raise Exception("invalid set name")
    net = load_test_model_from_factory(configs)
    rebuilder = Rebuilder(net, gpu_id=args.gpu_id)
    rebuilder.load_params(args.model_path)
    print('Model: {} has been loaded'.format(configs['model']['name']))

    # test each image
    print('Start Testing... ')
    if configs['db']['name'] == 'mvtec':
        threshold_seg_dict = val_mvtec(rebuilder, transform,configs)
        test_mvtec(test_set, rebuilder, transform, args.res_dir, threshold_seg_dict, configs)
    elif configs['db']['name'] == 'chip':
        test_chip(test_set, rebuilder, transform, args.res_dir, configs)
    else:
        raise Exception("Invalid set name")