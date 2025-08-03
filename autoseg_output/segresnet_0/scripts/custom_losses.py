import torch
from monai.losses import DiceLoss, HausdorffDTLoss
import torch.nn as nn

class DiceHausdorffLoss(nn.Module):
    def __init__(self, sigmoid=True):
        super().__init__()
        self.dice = DiceLoss(sigmoid=sigmoid)
        self.hausdorff = HausdorffDTLoss(sigmoid=sigmoid)
    def forward(self, input, target):
        return self.dice(input, target) + self.hausdorff(input, target)