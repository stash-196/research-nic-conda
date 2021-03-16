from model.parent_model import LitParentModel
import torch.nn as nn
import torch


class LitHidden1(LitParentModel):
    def __init__(self, *args):
        super(LitHidden1, self).__init__(*args)

        self.classifier = nn.Sequential(
            nn.Linear(self.perceptrons[0], self.perceptrons[1]),
            nn.ReLU(),
            nn.Linear(self.perceptrons[1], self.perceptrons[2]),
            nn.LogSoftmax(dim=1),
        )

    def training_step(self, train_batch, batch_idx):
        input, target = train_batch
        input = input.view(input.size(0), -1)
        output = self.classifier(input)
        loss = nn.NLLoss(target, output)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        y_hat = self.classifier(x)
        loss = nn.NLLLoss(y, y_hat)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.TRAIN.LR)
        return optimizer
