import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from model.graph_au_pre import *
from data_loader import *
from pre_process import *
from utils import *
from losses import *

def get_dataloader(conf_graph_au):
    print('==> Preparing data...')
    if conf_graph_au.dataset == 'BP4D':
        if conf_graph_au.info_nce == None:
            trainset = BP4D(root_path=conf_graph_au.dataset_path, train = True, info_nce = conf_graph_au.info_nce, 
                            fold = conf_graph_au.fold, transform = image_train_enhance(crop_size=conf_graph_au.crop_size), target_transform = None,  
                            crop_size = conf_graph_au.crop_size)
            train_loader = DataLoader(trainset, batch_size = conf_graph_au.batch_size, shuffle = True, num_workers = conf_graph_au.num_workers, pin_memory=True)
            valset = BP4D(root_path=conf_graph_au.dataset_path, train = False, info_nce = conf_graph_au.info_nce, 
                          fold = conf_graph_au.fold, transform = image_test(crop_size=conf_graph_au.crop_size), target_transform = None,
                          crop_size = conf_graph_au.crop_size)
            val_loader = DataLoader(valset, batch_size = conf_graph_au.batch_size, shuffle = False, num_workers = conf_graph_au.num_workers, pin_memory=True)
            return train_loader, val_loader, len(trainset), len(valset)
        
        elif conf_graph_au.info_nce == 'enhance':
            trainset = BP4D(root_path=conf_graph_au.dataset_path, train = True, info_nce = conf_graph_au.info_nce, 
                            fold = conf_graph_au.fold, transform = image_train_enhance(crop_size=conf_graph_au.crop_size), 
                            target_transform = image_train_enhance(crop_size=conf_graph_au.crop_size), crop_size = conf_graph_au.crop_size)
            train_loader = DataLoader(trainset, batch_size = conf_graph_au.batch_size, shuffle = True, num_workers = conf_graph_au.num_workers, pin_memory=True)
            return train_loader, len(trainset)
        
    elif conf_graph_au.dataset == 'DISFA':
        if conf_graph_au.info_nce == None:
            trainset = DISFA(root_path=conf_graph_au.dataset_path, train = True, info_nce = conf_graph_au.info_nce, 
                            fold = conf_graph_au.fold, transform = image_train_enhance(crop_size=conf_graph_au.crop_size), target_transform = None,  
                            crop_size = conf_graph_au.crop_size)
            train_loader = DataLoader(trainset, batch_size = conf_graph_au.batch_size, shuffle = True, num_workers = conf_graph_au.num_workers, pin_memory=True)
            valset = DISFA(root_path=conf_graph_au.dataset_path, train = False, info_nce = conf_graph_au.info_nce, 
                          fold = conf_graph_au.fold, transform = image_test(crop_size=conf_graph_au.crop_size), target_transform = None,
                          crop_size = conf_graph_au.crop_size)
            val_loader = DataLoader(valset, batch_size = conf_graph_au.batch_size, shuffle = False, num_workers = conf_graph_au.num_workers, pin_memory=True)
            return train_loader, val_loader, len(trainset), len(valset)
        
        elif conf_graph_au.info_nce == 'enhance':
            trainset = DISFA(root_path=conf_graph_au.dataset_path, train = True, info_nce = conf_graph_au.info_nce, 
                            fold = conf_graph_au.fold, transform = image_train_enhance(crop_size=conf_graph_au.crop_size), 
                            target_transform = image_train_enhance(crop_size=conf_graph_au.crop_size), crop_size = conf_graph_au.crop_size)
            train_loader = DataLoader(trainset, batch_size = conf_graph_au.batch_size, shuffle = True, num_workers = conf_graph_au.num_workers, pin_memory=True)
            return train_loader, len(trainset)

# Train
def train_graphau(net, train_loader, optimizer, epoch, train_weight):       
    net.train()
    losses = AverageMeter() 
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.float(), labels.float()
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs) 
        loss = au_softmax_loss(outputs, labels, train_weight)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
        if batch_idx % 10 == 0:
            print("epoch: {}  batch_idx: {}  loss_avg: {:.5f}".format(epoch, batch_idx, losses.avg))
    return losses.avg

def train_graphau_nce(net, train_loader, optimizer, epoch, train_weight):      
    net.train()
    losses = AverageMeter() 
    for batch_idx, (inputs, inputs_e, labels) in enumerate(tqdm(train_loader)):
        inputs, inputs_e, labels = inputs.float(), inputs_e.float(), labels.float()
        if torch.cuda.is_available():
            inputs, inputs_e, labels = inputs.cuda(), inputs_e.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs) 
        outputs_e = net(inputs_e) 
        loss = au_infonce_loss(outputs, outputs_e, labels, train_weight)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
        if batch_idx % 10 == 0:
            print("epoch: {}  batch_idx: {}  loss_avg: {:.5f}".format(epoch, batch_idx, losses.avg))
    return losses.avg

# Val
def val_graphau(net, val_loader, test_weight):
    net.eval()
    losses = AverageMeter()
    statistics_list = None
    for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, labels = inputs.float(), labels.float()
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs) 
            loss = au_softmax_loss(outputs, labels, test_weight)
            losses.update(loss.data.item(), inputs.size(0))   
            update_list = statistics(outputs[:, 0, :], labels.detach(), 0.5)  
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list

def main(conf_graph_au):
    if conf_graph_au.dataset == 'BP4D':
        dataset_info = BP4D_infolist
    elif conf_graph_au.dataset == 'DISFA':
        dataset_info = DISFA_infolist

    # data
    if conf_graph_au.info_nce == None: 
        train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf_graph_au)
    elif conf_graph_au.info_nce == 'enhance':
        train_loader, train_data_num = get_dataloader(conf_graph_au)
    train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf_graph_au.dataset_path, 'list', conf_graph_au.dataset+'_train_weight_fold' + str(conf_graph_au.fold)+'.txt'))).cuda()
    test_weight = torch.from_numpy(np.loadtxt(os.path.join(conf_graph_au.dataset_path, 'list', conf_graph_au.dataset+'_test_weight_fold' + str(conf_graph_au.fold)+'.txt'))).cuda()
    logging.info("Fold: [{} | {}]".format(conf_graph_au.fold, conf_graph_au.N_fold))

    if conf_graph_au.info_nce == None: 
        net = Graph_au_net(au_num_classes=conf_graph_au.au_num_classes, backbone=conf_graph_au.backbone, 
                           neighbor_num=conf_graph_au.neighbor_num, metric=conf_graph_au.metric).cuda()
        for p in net.parameters():
            p.requires_grad = True
    elif conf_graph_au.info_nce == 'enhance':
        net = Graph_au_net_NCE(au_num_classes=conf_graph_au.au_num_classes, backbone=conf_graph_au.backbone, 
                               neighbor_num=conf_graph_au.neighbor_num, metric=conf_graph_au.metric).cuda()
        for p in net.parameters():
            p.requires_grad = True

    if conf_graph_au.start_epoch > 0: 
        logging.info("Resume form epoch {} ".format(conf_graph_au.start_epoch))
        if conf_graph_au.info_nce == None: 
            net.load_state_dict(torch.load(os.path.join(conf_graph_au['outdir'], 'epoch' + str(conf_graph_au.start_epoch) + '_graphau_net_model_fold' + str(conf_graph_au.fold) + '.pth')))
        elif conf_graph_au.info_nce == 'enhance':
            net.load_state_dict(torch.load(os.path.join(conf_graph_au['outdir'], 'epoch' + str(conf_graph_au.start_epoch) + '_graphau_net_nce_model_fold' + str(conf_graph_au.fold) + '.pth')))

    optimizer = torch.optim.AdamW(net.parameters(), lr=conf_graph_au.init_lr, betas=(0.9, 0.999), weight_decay = conf_graph_au.weight_decay)
    print('the init learning rate is ', conf_graph_au.init_lr)    

    for epoch in range(conf_graph_au.start_epoch + 1, conf_graph_au.n_epochs + 1): 
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch, conf_graph_au.n_epochs, lr))
        if conf_graph_au.info_nce == None: 
            train_loss = train_graphau(net, train_loader, optimizer, epoch, train_weight)
            val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val_graphau(net, val_loader, test_weight)
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
            torch.save(net.state_dict(), os.path.join(conf_graph_au['outdir'], 'epoch' + str(epoch) + '_graphau_net_model_fold' + str(conf_graph_au.fold) + '.pth'))

        elif conf_graph_au.info_nce == 'enhance': 
            train_loss = train_graphau_nce(net, train_loader, optimizer, epoch, train_weight)
            infostr = {'Epoch: {}   train_loss: {:.5f}'.format(epoch, train_loss)}
            logging.info(infostr)
            torch.save(net.state_dict(), os.path.join(conf_graph_au['outdir'], 'epoch' + str(epoch) + '_graphau_net_nce_model_fold' + str(conf_graph_au.fold) + '.pth'))            

if __name__=="__main__":
    conf_graph_au = get_config()
    set_env(conf_graph_au)
    set_outdir(conf_graph_au)
    set_logger(conf_graph_au)
    main(conf_graph_au)
