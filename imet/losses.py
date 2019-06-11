from torch import nn
import torch


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = nn.functional.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1)



class FBeta(nn.Module):
    def __init__(self, threshold=0.1, beta=2, reduction='mean'):
        super().__init__()
        self.threshold = threshold
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
       
        loss = 0
        lack_cls = target.sum(dim=0) == 0
        if lack_cls.any():
            loss += nn.functional.binary_cross_entropy_with_logits(
                input, target, reduction=self.reduction)
        predict = torch.sigmoid(input)
        predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
        tp = predict * target
        tp = tp.sum(dim=0)
        precision = tp / (predict.sum(dim=0) + 1e-8)
        recall = tp / (target.sum(dim=0) + 1e-8)
        f1 = (1 + self.beta ** 2) * (precision * recall / (self.beta ** 2 * precision + recall + 1e-8))
        if self.reduction == 'none':
            loss = loss.mean(dim=0)
        if self.reduction == 'mean':
            f1 = f1.mean()
        return 1 - f1 + loss



losses_dict = {
    'bce' :  nn.BCEWithLogitsLoss(reduction='none'),
    'focal': FocalLoss(),
    'fbeta': FBeta(reduction='none')
}


def loss_function(loss, pos_weight=None):
    criterion = losses_dict[loss]
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    return criterion