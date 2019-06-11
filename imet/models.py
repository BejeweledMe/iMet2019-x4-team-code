from torch import nn
from torch.nn import functional as F
from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def get_model(model: str, num_classes: int, pretrained: bool, input_size: int, use_cuda: bool=None) -> nn.Module:
    if model.startswith('efficientnet'):
        model = EfficientNet.from_name(model)
        model._fc = nn.Linear(1536, num_classes)
    else:    
        model = make_model(model, num_classes, pretrained, input_size=(input_size, input_size), pool=AvgPool())
    if use_cuda:
        model = model.cuda()
    return model