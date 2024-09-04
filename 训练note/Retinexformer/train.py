import argparse
import datetime
import logging
import math
import os

import random
import time
import torch
from os import path as osp

# 创建数据加载器和数据集
from basicsr.data import create_dataloader, create_dataset
# 用于在数据记载过程中进行采样操作
from basicsr.data.data_sampler import EnlargedSampler
# 负责在 CPU 和 CUDA 之间进行数据预取操作，以提高数据加载效率。
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
# 用于创建深度学习模型。
from basicsr.models import create_model
# MessageLogger: 处理日志记录的类，便于跟踪训练过程。
# check_resume: 用于检查并恢复训练状态
# 一些常用的实用函数，分别用于获取环境信息、初始化日志记录器、获取时间戳、初始化 TensorBoard 记录器、
# 初始化 WandB 记录器、创建实验目录和设置随机种子。

from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
# 用于分布式训练的初始化和获取分布式训练信息
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.misc import mkdir_and_rename2
# 用于将字典转换为字符串格式和解析 YAML 配置文件
from basicsr.utils.options import dict2str, parse

import numpy as np

from pdb import set_trace as stx

def parse_options(is_train=True):
    # 创建一个参数解析器对象，用于从命令行读取输入参数
    parser = argparse.ArgumentParser()
    # parser.add_argument(): 为命令行参数添加选项
    # --opt: YAML 配置文件的路径
    parser.add_argument(
        '--opt', type=str, default='Options/RetinexFormer_LOL_v1.yml', help='Path to option YAML file.')
    # --launcher: 指定作业启动器的类型，如 none、pytorch 或 slurm
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    # --local_rank: 本地进程的 rank，用于分布式训练
    parser.add_argument('--local_rank', type=int, default=0)
    # 解析命令行参数，并将其存储在 args 对象中
    args = parser.parse_args()
    # 使用 basicsr 提供的 parse 函数解析 YAML 文件，并返回配置选项的字典
    opt = parse(args.opt, is_train=is_train)

    
    # distributed settings
    # args.launcher == 'none': 判断是否启用分布式训练。如果不使用分布式训练 ('none')，则设置 opt['dist'] = False
    if args.launcher == 'none':
        
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)
            
    # opt['rank'], opt['world_size'] = get_dist_info(): 
    # 获取当前进程的 rank 和 world size，这些参数在分布式训练中非常重要，用于标识不同的进程
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed 随机种子
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt

# 用于初始化训练过程中的日志记录器
def init_loggers(opt): # 接受参数 opt，这是从配置文件中解析出来的配置选项
    # 设置训练日志文件路径
    
    # 用于拼接路径；opt['path']['log'] 指定了日志文件的目录
    # f"train_{opt['name']}_{get_time_str()}.log" 是日志文件的名称，包含了当前训练的名称和时间戳
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    # 初始化主日志记录器
    # get_root_logger(): 这是从 basicsr 中导入的一个函数，用于获取根日志记录器。
    # 它会创建或获取一个名为 'basicsr' 的日志记录器，并设置日志级别为 INFO，将日志输出到 log_file 中
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    
    # 这里是为指标记录创建一个新的日志文件路径，文件名为 metric.csv，将用于记录训练过程中的指标数据
    log_file = osp.join(opt['path']['log'],
                        f"metric.csv")
    # 创建一个新的日志记录器，记录训练中的指标信息，日志级别设置为 INFO，输出到 metric.csv 文件中
    logger_metric = get_root_logger(logger_name='metric',
                                    log_level=logging.INFO, log_file=log_file)
    
    # 记录初始的指标名称
    # 初始化一个字符串metric_str，其中包含了当前时间
    metric_str = f'iter ({get_time_str()})'
    # 遍历配置文件中验证阶段的所有指标
    for k, v in opt['val']['metrics'].items():
        # 将每个指标的名称添加到 metric_str 中
        metric_str += f',{k}'
    # 将构建好的 metric_str 写入指标日志文件中，作为 CSV 文件的表头
    logger_metric.info(metric_str)

    # 获取并记录当前环境的信息
    logger.info(get_env_info())
    # 将配置选项字典 opt 转换为字符串并记录下来
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    # if (opt['logger'].get('wandb')
    #         is not None) and (opt['logger']['wandb'].get('project')
    #                           is not None) and ('debug' not in opt['name']):
    #     assert opt['logger'].get('use_tb_logger') is True, (
    #         'should turn on tensorboard when using wandb')
    #     init_wandb_logger(opt)

    # 初始化tensorBoard记录器
    # 初始化 TensorBoard 记录器变量
    tb_logger = None
    # 检查是否启用了 TensorBoard 记录器，并且当前模式不是调试模式
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # 如果条件满足，初始化 TensorBoard 记录器，并将日志保存在 tb_logger/训练名称 目录下。
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
        # 返回主日志记录器和 TensorBoard 记录器。
    return logger, tb_logger

# 用于构建训练train和验证val的数据加载器
def create_train_val_dataloader(opt, logger):  #train loader 和 val loader 一起构建
    # opt: 包含配置文件中的所有设置的字典。
    # logger: 用于记录日志信息的日志记录器
    
    # create train and val dataloaders
    train_loader, val_loader = None, None # 分别用于存储训练和验证的数据加载器
    
    # 遍历配置文件中 datasets 部分的每个数据集配置
    # phase 代表当前数据集的阶段（train 或 val），dataset_opt 是该阶段的具体配置。
    for phase, dataset_opt in opt['datasets'].items():
        # stx()
        
        # 检查当前数据集是否为训练数据集
        if phase == 'train':
            
            # 获取数据集增广比
            # 从配置文件中获取 dataset_enlarge_ratio 参数，表示数据集的增广比（如果未指定，默认为 1）
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            
            # 创建训练数据集
            # 调用 create_dataset 函数，传入训练数据集的配置参数 dataset_opt，创建训练数据集 train_set
            train_set = create_dataset(dataset_opt)    #将option中的dataset参数传入create_dataset中构建train_set
            
            # stx()
            # 创建数据采样器
            # EnlargedSampler: 是一个用于数据采样的类，创建一个训练数据的采样器
            # train_sampler。采样器的作用是在分布式训练中，根据 GPU 的数量（world_size）和每个 GPU 的索引（rank）合理分配数据
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)     #缺少关键字 world_size 和 rank，train_sampler是做什么？从get_dist_info得到
            
            # stx()
            # 创建训练数据加载器
            # 调用 create_dataloader 函数，使用训练数据集 train_set 和数据采样器 train_sampler，以及其他相关配置参数，创建训练数据加载器 train_loader
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
            # stx()
            
            # 计算每个 epoch 的迭代次数和总的 epoch 数
            # per_epoch计算每个 epoch 需要的迭代次数
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))  #一个epoch遍历一次数据
            # 从配置文件中读取训练的总迭代次数
            total_iters = int(opt['train']['total_iter'])
            # 计算总的 epoch 数，并向上取整
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch)) #一个iteration就是一次 inference + backward，总的iteration是不变的
            # 记录训练统计信息
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            # stx()
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name']) #状态路径
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0: #如果路径已存在
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary，resume_state是重新训练的时候接上的吗？
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename2(
                osp.join('tb_logger', opt['name']), opt['rename_flag'])

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        best_metric = resume_state['best_metric']
        best_psnr = best_metric['psnr']
        best_iter = best_metric['iter']
        logger.info(f'best psnr: {best_psnr} from iteration {best_iter}')
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
        best_metric = {'iter': 0}
        for k, v in opt['val']['metrics'].items():
            best_metric[k] = 0
        # stx()

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch

    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        
        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # ------Progressive learning ---------------------
            j = ((current_iter > groups) != True).nonzero()[
                0]  # 根据当前的iter次数判断在哪个阶段
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(
                    mini_gt_size, mini_batch_size * torch.cuda.device_count()))
                logger_j[bs_j] = False

            lq = train_data['lq']
            gt = train_data['gt']

            if mini_batch_size < batch_size:  # 默认生成batch_size对图片，小于就要抽样
                indices = random.sample(
                    range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
            # -------------------------------------------
            # print(lq.shape)
            model.feed_train_data({'lq': lq, 'gt': gt})
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter, best_metric=best_metric)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                current_metric = model.validation(val_loader, current_iter, tb_logger,
                                                  opt['val']['save_img'], rgb2bgr, use_image)
                # log cur metric to csv file
                logger_metric = get_root_logger(logger_name='metric')
                metric_str = f'{current_iter},{current_metric}'
                logger_metric.info(metric_str)

                # log best metric
                if best_metric['psnr'] < current_metric:
                    best_metric['psnr'] = current_metric
                    # save best model
                    best_metric['iter'] = current_iter
                    model.save_best(best_metric)
                if tb_logger:
                    tb_logger.add_scalar(  # best iter
                        f'metrics/best_iter', best_metric['iter'], current_iter)
                    for k, v in opt['val']['metrics'].items():  # best_psnr
                        tb_logger.add_scalar(
                            f'metrics/best_{k}', best_metric[k], current_iter)

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()