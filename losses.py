import torch

def au_softmax_loss(input, target, weight, size_average=True): 
    t_input_pos = input[:, 0] 
    t_input_neg = input[:, 1] 
    los_pos = target * torch.log(t_input_pos.clamp(min=1e-3)) 
    los_neg = (1 - target) * torch.log(t_input_neg.clamp(min=1e-3)) 
    t_loss = -(los_pos + los_neg)
    if weight is not None:
        t_loss = t_loss * weight
    if size_average:
        return t_loss.mean()
    else:
        return t_loss.sum()
    
def au_infonce_loss(input, input_e, target, weight, temperature = 0.5, size_average=True): 
    out = torch.cat([input, input_e], dim=0) 
    t_target = torch.cat([target, target], dim=0) 
    t_loss = torch.zeros([t_target.size(0), t_target.size(1)],dtype=torch.float) 
    scores = torch.exp((torch.einsum('i j k,j l k -> i l k', out, torch.transpose(out, 0, 1).contiguous())/temperature)) 
    labels = t_target.unsqueeze(0) 
    anchor_labels = t_target.reshape(-1,1,input.size(2)) 
    occur_labels = (t_target == 1).float() 
    unoccur_labels = (t_target == 0).float() 
    p = torch.rand([input.size(2)]).cuda() 

    relation_pos = (labels == anchor_labels).float()    
    pos_scores_max, _ = (relation_pos * scores).max(dim=-2) 
    pos_scores_max = pos_scores_max * (p < 0.2).float() 
    
    pos_scores_mean = (relation_pos * scores).mean(dim=-2) 
    pos_scores_mean = pos_scores_mean * (p >= 0.2).float() * (p < 0.8).float() 

    pos_scores_ori_eye = torch.exp(torch.sum(input * input_e, dim=-2)/temperature)  
    pos_scores_ori = torch.cat([pos_scores_ori_eye, pos_scores_ori_eye], dim=0) 
    pos_scores_ori = pos_scores_ori * (p >= 0.8).float() 

    pos_scores = pos_scores_max + pos_scores_mean + pos_scores_ori

    relation_neg = (labels != anchor_labels).float() 
    neg_scores = (relation_neg * scores ** 1.2).sum(dim=-2) * occur_labels + (relation_neg * scores ** 0.4).sum(dim=-2) * unoccur_labels 
    t_loss = -(torch.log((pos_scores)/(pos_scores + neg_scores))).mean(dim=-2) 
    if weight is not None:                 
        t_loss = t_loss * weight

    if size_average:
        return t_loss.mean()
    else:
        return t_loss.sum()    
