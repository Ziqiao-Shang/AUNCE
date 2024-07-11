import argparse
from easydict import EasyDict as edict

def str2bool(v):     
    return v.lower() in ('true') 

parser = argparse.ArgumentParser(description='PyTorch Training')
# Datasets
parser.add_argument('--dataset', type=str, default="BP4D", help="experiment dataset BP4D / DISFA")
parser.add_argument('--N_fold', type=int, default=1, help="the ratio of train and validation data")
parser.add_argument('--fold', type=int, default=3, metavar='N', help='the fold of three folds cross-validation ')

# Mode setting
parser.add_argument('--info_nce', type=str, default=None, help="The strategy to use InfoNCE for training") 
parser.add_argument('--crop_size', type=int, default=224, help="crop size of train/test image data")
parser.add_argument('--unit_dim', type=int, default=4, help='unit dims') 

# Training Param
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--init_lr', type=float, default=1e-5, metavar='LR', help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch') 
parser.add_argument('--n_epochs', type=int, default=60, metavar='N', help='number of total epochs to run')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay for optimizer')
parser.add_argument('--backbone', type=str, default='swin_transformer_base', help='backbone type') 
parser.add_argument('--neighbor_num', type=int, default=4, help='neighbor nums')
parser.add_argument('--metric', type=str, default='dots', help='metric generation')                                                   
parser.add_argument('--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

# Other Settings
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--exp_name', type=str, default="Graphau_bp4d_swin_fold_1", help="experiment name for saving checkpoints")

def parser2dict():
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)
