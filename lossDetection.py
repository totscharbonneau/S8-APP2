from torch import Tensor
from torch import nn

class LossDetection():
    def forward(self, input: Tensor, target: Tensor) -> Tensor:


        objectpresenceLoss = nn.BCELoss(input[:, 1], target[:, 1])
        positionLoss = nn.MSELoss(input[:, 1:4], target[:, 1:4])

        classLoss = nn.CrossEntropyLoss(input[:,5:7],target[:,5])

        totalLoss = objectpresenceLoss + positionLoss + classLoss

        return totalLoss


