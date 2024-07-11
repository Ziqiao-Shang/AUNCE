import torch
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import os
import numpy as np
import torch
import logging
import yaml
from datetime import datetime
import conf_graph_au

class AverageMeter(object): 
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def statistics(pred, y, thresh): 
    batch_size = pred.size(0)
    class_nb = pred.size(1)
    pred = pred >= thresh
    pred = pred.long()
    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(batch_size):
            if pred[i][j] == 1:
                if y[i][j] == 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    assert False
            elif pred[i][j] == 0:
                if y[i][j] == 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    assert False
            else:
                assert False
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list

def calc_f1_score(statistics_list):
    f1_score_list = []
    for i in range(len(statistics_list)): 
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        precise = TP / (TP + FP + 1e-20) 
        recall = TP / (TP + FN + 1e-20)  
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)
    return mean_f1_score, f1_score_list

def calc_acc(statistics_list):
    acc_list = []
    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        TN = statistics_list[i]['TN']
        acc = (TP+TN)/(TP+TN+FP+FN)
        acc_list.append(acc)
    mean_acc_score = sum(acc_list) / len(acc_list)
    return mean_acc_score, acc_list

def update_statistics_list(old_list, new_list): 
    if not old_list:
        return new_list
    assert len(old_list) == len(new_list)
    for i in range(len(old_list)):
        old_list[i]['TP'] += new_list[i]['TP']
        old_list[i]['FP'] += new_list[i]['FP']
        old_list[i]['TN'] += new_list[i]['TN']
        old_list[i]['FN'] += new_list[i]['FN']
    return old_list

def BP4D_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU17: {:.2f} AU23: {:.2f} AU24: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9],100.*list[10],100.*list[11])}
    return infostr

def DISFA_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU9: {:.2f} AU12: {:.2f}  AU25: {:.2f} AU26: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7])}
    return infostr

def str2bool(v):
    return v.lower() in ('true', '1')

def print_conf(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()): 
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message

def get_config():
    cfg = conf_graph_au.parser2dict()
    if cfg.dataset == 'BP4D':
        with open('config/BP4D_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)   
            datasets_cfg = edict(datasets_cfg) 
    elif cfg.dataset == 'DISFA':
        with open('config/DISFA_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
    else:
        raise Exception("Unkown Datsets:",cfg.dataset)

    cfg.update(datasets_cfg) 
    return cfg

def set_env(cfg):
    if 'cudnn' in cfg:
        torch.backends.cudnn.benchmark = cfg.cudnn
    else:
        torch.backends.cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

def set_outdir(conf):
    default_outdir = 'results'
    if 'timedir' in conf:
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p')
        outdir = os.path.join(default_outdir,conf.exp_name,timestr)
    else:
        outdir = os.path.join(default_outdir,conf.exp_name)
        prefix = 'bs_'+str(conf.batch_size)+'_seed_'+str(conf.seed)+'_lr_'+str(conf.init_lr)
        outdir = os.path.join(outdir,prefix)
    ensure_dir(outdir)
    conf.outdir = outdir
    return conf

def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('{} is created'.format(dir_name))


def set_logger(cfg):
    if 'loglevel' in cfg:
        loglevel = eval('logging.'+loglevel)
    else:
        loglevel = logging.INFO
    outname = 'train.log'

    outdir = cfg['outdir']
    log_path = os.path.join(outdir,outname)

    logger = logging.getLogger()
    logger.setLevel(loglevel)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logging.info(print_conf(cfg))
    logging.info('writting logs to file {}'.format(log_path))

def set_linear(cfg):
    default_outdir = 'results'
    outdir = os.path.join(default_outdir,cfg.exp_name)
    prefix = 'bs_'+str(cfg.batch_size)+'_seed_'+str(cfg.seed)+'_lr_'+str(cfg.init_lr)
    outdir = os.path.join(outdir, prefix)
    ensure_dir(outdir)
    loglevel = logging.INFO
    outname = 'linear_test.log'
    log_path = os.path.join(outdir,outname)
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    logging.info(print_conf(cfg))
    logging.info('writting logs to file {}'.format(log_path))
