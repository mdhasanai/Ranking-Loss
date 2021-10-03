import torch

def ranking_lossT(logitsT, labelsT):
    """
    logitsT: is the output embeddings where shape is (b,e), b is the batch and e is the embedings shape, eg: (128,512)
    labelsT: onehot embedding of labels
    """
    eps = 1e-8
    subset_idxT = torch.sum(torch.abs(labelsT),dim=0)
    subset_idxT = (subset_idxT>0).nonzero().view(-1).long().cuda()
    sub_labelsT = labelsT[:,subset_idxT]
    sub_logitsT = logitsT[:,subset_idxT]    
    positive_tagsT = torch.clamp(sub_labelsT,0.,1.)
    negative_tagsT = torch.clamp(-sub_labelsT,0.,1.)
    maskT = positive_tagsT.unsqueeze(1) * negative_tagsT.unsqueeze(-1)
    pos_score_matT = sub_logitsT * positive_tagsT
    neg_score_matT = sub_logitsT * negative_tagsT
    IW_pos3T = pos_score_matT.unsqueeze(1)
    IW_neg3T = neg_score_matT.unsqueeze(-1)
    OT = 1 + IW_neg3T - IW_pos3T
    O_maskT = maskT * OT
    diffT = torch.clamp(O_maskT, 0)
    violationT = torch.sign(diffT).sum(1).sum(1) 
    diffT = diffT.sum(1).sum(1) 
    lossT =  torch.mean(diffT / (violationT+eps))
    return lossT
