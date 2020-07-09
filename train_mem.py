import os
import json
import argparse
from db import training_collate
from tools import Timer, Log
from factory import *
from datetime import datetime
from torch.utils import data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Name of .json file', type=str, required=True)
    parser.add_argument('--ngpu', help='Numbers of GPU', type=int, default=1)
    parser.add_argument('--log_dir', help="Directory of training log", type=str, default='./log')
    args = parser.parse_args()

    return args


def adjust_learning_rate(trainer, init_lr, decay_rate, epoch, step_index, iteration, epoch_size):
    if epoch < 6:
        lr = 1e-6 + (init_lr-1e-6) * iteration / (epoch_size * 5)
    else:
        lr = init_lr / (decay_rate ** (step_index))

    trainer.set_lr(lr)

    return lr


if __name__ == '__main__':
    args = parse_args()
    DATETIME = datetime.strftime(datetime.now(), '%Y-%m%d-%H%M')

    # load config
    cfg_file = os.path.join('./config', args.cfg)
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    start_epoch = configs['op']['start_epoch']
    max_epoch = configs['op']['max_epoch']
    learning_rate = configs['op']['learning_rate']
    decay_rate = configs['op']['decay_rate']
    epoch_steps = configs['op']['epoch_steps']
    snapshot = configs['op']['snapshot']
    batch_size = configs['db']['batch_size']
    loader_threads = configs['db']['loader_threads']
    save_dir = configs['system']['save_dir']
    save_dir += '_{}'.format(DATETIME)

    # init Timer
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    _t = Timer()

    # create log file
    log = Log(args.log_dir, args.cfg)
    log.wr_cfg(configs)

    # load data set
    training_set = load_data_set_from_factory(configs, 'train')
    print('Data set: {} has been loaded'.format(configs['db']['name']))

    # load model
    trainer = load_training_model_from_factory(configs, ngpu=args.ngpu)
    if configs['system']['resume']:
        trainer.load_params(configs['system']['resume_path'])
    print('Model: {} has been loaded'.format(configs['model']['name']))

    train_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                   pin_memory=True)
    # start training
    print('Start training...')
    iteration = 0
    epoch_size = len(training_set) // batch_size
    for epoch in range(max_epoch):
        # save parameters
        if epoch % snapshot == 0 and epoch > 0:
            save_name = '{}-{:d}.pth'.format(args.cfg, epoch)
            save_path = os.path.join(save_dir, save_name)
            trainer.save_params(save_path)
        epoch += 1

        # adjust learning rate
        step_index = 0
        for step in epoch_steps:
            if epoch >= step:
                step_index += 1
        lr = adjust_learning_rate(trainer, learning_rate, decay_rate, epoch, step_index, iteration, epoch_size)

        for seq, sample in enumerate(train_loader):
            iteration += 1
            # load data
            _t.tic()
            normals, query, target = sample.values()
            if configs['model']['type'] == 'MEM':
                trainer.train(normals, query, target)
            else:
                raise Exception("Wrong model type!")
            batch_time = _t.toc()

            # print message
            if iteration % 10 == 0:
                _t.clear()
                mes = 'Epoch:' + repr(epoch) + '||epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                mes += '||Totel iter: ' + repr(iteration)
                mes += '||{}'.format(trainer.get_loss_message())
                mes += '||LR: %.8f' % (lr)
                mes += '||Batch time: %.4f sec.' % batch_time
                log.wr_mes(mes)
                print(mes)
    save_name = '{}-{:d}.pth'.format(args.cfg, epoch)
    save_path = os.path.join(save_dir, save_name)
    trainer.save_params(save_path)
    log.close()
    exit(0)

