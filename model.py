import torch
import torch.nn as nn
import torchvision.models as models
from copy import deepcopy

class BirdModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(BirdModel, self).__init__()

        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)
        self.in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2), 
            nn.Linear(self.in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)