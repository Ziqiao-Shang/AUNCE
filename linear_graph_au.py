import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from data_loader import *
from pre_process import *
from utils import *
from losses import *
from model.graph_au_pre import *

parser = argparse.ArgumentParser(description='Linear Evaluation')

parser.add_argument('--dataset', type=str, default="BP4D", help="experiment dataset BP4D / DISFA")
parser.add_argument('--fold', type=int, default=1, metavar='N', help='the fold of three folds cross-validation ')
parser.add_argument('--N_fold', type=int, default=3, help="the ratio of train and validation data")

# Mode setting
parser.add_argument('--crop_size', type=int, default=224, help="crop size of train/test image data")
parser.add_argument('--au_num_classes', type=int, default=12, help="crop size of train/test image data")
parser.add_argument('--unit_dim', type=int, default=4, help='unit dims') 

# Training Param
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--init_lr', type=float, default=0.001, metavar='LR', help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch') 
parser.add_argument('--n_epochs', type=int, default=100, metavar='N', help='number of total epochs to run') 
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay for optimizer')                                                  
parser.add_argument('--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

# Experiment
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--dataset_path', type=str, default="data/BP4D", help="experiment dataset path of BP4D / DISFA")
parser.add_argument('--model_path', type=str, default='good_pretrain_model/Graphau_bp4d_swin_nce_step_1/bp4d_graphau_net_nce_model_fold1.pth', help='The pretrained model path')
parser.add_argument('--exp_name', type=str, default="Linear_graphau_bp4d_swin_nce_step_1", help="experiment name for saving checkpoints")

args = parser.parse_args()

def main(args):
    if args.dataset == 'BP4D':
        dataset_info = BP4D_infolist
    elif args.dataset == 'DISFA':
        dataset_info = DISFA_infolist

    # data
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(args)
    train_weight = torch.from_numpy(np.loadtxt(os.path.join(args.dataset_path, 'list', args.dataset+'_train_weight_fold' + str(args.fold) + '.txt'))).cuda()
    test_weight = torch.from_numpy(np.loadtxt(os.path.join(args.dataset_path, 'list', args.dataset+'_test_weight_fold' + str(args.fold) + '.txt'))).cuda()
    logging.info("Fold: [{} | {}]".format(args.fold, args.N_fold))

    val_net = Val_Net(au_num_classes = args.au_num_classes, unit_dim = args.unit_dim).cuda()
    for p in val_net.f.parameters():
        p.requires_grad = False

    if args.start_epoch > 0: 
        logging.info("Resume form epoch {} ".format(args.start_epoch))
        linear_outdir = os.path.join('results', 'linear_eval')
        ensure_dir(linear_outdir)
        val_net.load_state_dict(torch.load(os.path.join(linear_outdir, 'epoch' + str(args.start_epoch) + '_val_net_model_fold' + str(args.fold) + '.pth')))

    optimizer = torch.optim.Adam(val_net.fc.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    print('the init learning rate is ', args.init_lr)

    #train and val
    for epoch in range(args.start_epoch + 1, args.n_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch, args.n_epochs, lr))
        train_loss = train(val_net, train_loader, optimizer, epoch, train_weight)
        val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(val_net, val_loader, test_weight)
        # log
        infostr = {'Epoch: {}  train_loss: {:.5f}  val_loss: {:.5f}  val_mean_f1_score {:.2f}  val_mean_acc {:.2f}'
                .format(epoch, train_loss, val_loss, 100.* val_mean_f1_score, 100.* val_mean_acc)}
        logging.info(infostr)
        infostr = {'F1-score-list:'}
        logging.info(infostr)
        infostr = dataset_info(val_f1_score)
        logging.info(infostr)
        infostr = {'Acc-list:'}
        logging.info(infostr)
        infostr = dataset_info(val_acc)
        logging.info(infostr)

        linear_outdir = os.path.join('results', 'linear_eval')
        ensure_dir(linear_outdir)
        torch.save(val_net.state_dict(), os.path.join(linear_outdir, 'epoch' + str(epoch) + '_val_net_model_fold' + str(args.fold) + '.pth'))

def get_dataloader(args):
    print('==> Preparing data...')
    if args.dataset == 'BP4D':
        trainset = BP4D(root_path=args.dataset_path, train = True, info_nce = None, 
                            fold = args.fold, transform = image_train_enhance(crop_size=args.crop_size), target_transform = None,  
                            crop_size = args.crop_size)
        train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, pin_memory=True)
        valset = BP4D(root_path=args.dataset_path, train = False, info_nce = None, 
                        fold = args.fold, transform = image_test(crop_size=args.crop_size), target_transform = None,
                        crop_size = args.crop_size)
        val_loader = DataLoader(valset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory=True)
        return train_loader, val_loader, len(trainset), len(valset)
            
    elif args.dataset == 'DISFA':
        trainset = DISFA(root_path=args.dataset_path, train = True, info_nce = None, 
                            fold = args.fold, transform = image_train(crop_size=args.crop_size), target_transform = None,  
                            crop_size = args.crop_size)
        train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, pin_memory=True)
        valset = DISFA(root_path=args.dataset_path, train = False, info_nce = None, 
                        fold = args.fold, transform = image_test(crop_size=args.crop_size), target_transform = None,
                        crop_size = args.crop_size)
        val_loader = DataLoader(valset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory=True)
        return train_loader, val_loader, len(trainset), len(valset)

def train(val_net, train_loader, optimizer, epoch, train_weight):      
    val_net.train()
    losses = AverageMeter() 
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.float(), labels.float()
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        # forward 
        outputs = val_net(inputs) 
        # backward
        loss = au_softmax_loss(outputs, labels, train_weight)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
        if batch_idx % 10 == 0:
            print("epoch: {}  batch_idx: {}  loss_avg: {:.5f}".format(epoch, batch_idx, losses.avg))
    return losses.avg

def val(val_net, val_loader, test_weight): 
    val_net.eval()
    losses = AverageMeter()
    statistics_list = None
    for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, labels = inputs.float(), labels.float()
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = val_net(inputs) 
            loss = au_softmax_loss(outputs, labels, test_weight) 
            losses.update(loss.data.item(), inputs.size(0))   
            update_list = statistics(outputs[:, 0, :], labels.detach(), 0.5)  
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list

class Val_Net(nn.Module):
    def __init__(self, au_num_classes, unit_dim=8):
        super(Val_Net, self).__init__()
        model = Graph_au_net_NCE(au_num_classes = args.au_num_classes, backbone='swin_transformer_base', neighbor_num=4, metric='dots').cuda()
        model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))
        self.f = model
        self.fc = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(unit_dim * 256, 2)                            
        ) for i in range(au_num_classes)])

    def forward(self, x):
        feature=self.f(x)
        for i in range(len(self.fc)):
            au_output = self.fc[i](feature[:,:,i])             
            au_output = au_output.unsqueeze(2) 
            au_output = torch.softmax(au_output, dim=1)
            if i == 0:
                aus_output = au_output
            else:
                aus_output = torch.cat((aus_output, au_output), dim=2)   
        return aus_output  

if __name__ == '__main__':
    set_seed(3407)
    set_env(args)
    set_linear(args)
    main(args)
