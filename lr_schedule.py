import math
def step_lr_scheduler(param_lr, optimizer, epoch, gamma, stepsize, init_lr=0.00001): 
    if epoch == 1:
        init_lr = param_lr
    lr = init_lr * (gamma ** (epoch // stepsize)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def cos_adjust_lr_scheduler(param_lr, optimizer, epoch, n_epochs, batch_idx, train_loader_len, init_lr=0.00001):
    current_iter = batch_idx + epoch * train_loader_len
    max_iter = n_epochs * train_loader_len
    if epoch == 1:
        init_lr = param_lr
    lr = init_lr * (1 + math.cos(math.pi * current_iter / max_iter)) / 2 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

schedule_dict = {'step': step_lr_scheduler, 'cos_adjust':cos_adjust_lr_scheduler}
