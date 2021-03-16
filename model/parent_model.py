from abc import abstractmethod
import torch.nn as nn
import pytorch_lightning as pl


class LitParentModel(pl.LightningModule):
    def __init__(self, cfg, device, perceptrons):
        self.cfg = cfg
        self.device = device
        self.perceptrons = perceptrons

    def get_parameter_tensors(self):
        params = []
        for layer in self.classifier:
            try:
                params.append(layer.weight)
                params.append(layer.bias)
            except:
                continue
        return params

    def forward(self, x):
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.get_parameter_tensors() if p.requires_grad)
