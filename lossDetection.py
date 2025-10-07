from torch import Tensor
from torch import nn
import torch
class LossDetection():

    def __call__(self,  input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input,target)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        mask = target[:, :, 0] > 0.5

        mask2 = mask.unsqueeze(-1)


        bceloss = nn.BCELoss()
        mseloss = nn.MSELoss(reduction='none')
        celoss = nn.CrossEntropyLoss()

        input1 = input[:, :, 0]
        target1 = target[:, :, 0]
        objectpresenceLoss = bceloss(input1, target1)

        if mask.sum() > 1:

            input2 = input[:, :, 1:4]
            target2 = target[:, :, 1:4]
            positionLoss = mseloss.forward(input2, target2)
            positionLoss = (positionLoss * mask2).sum()/ mask.sum()


            input3 = input[:, :,4:7][mask]
            target3 = target[:, :, 4][mask]

            classLoss = celoss.forward(input3,target3.long())

        else:
            positionLoss = 0
            classLoss = 0
        # print(f'Class: {classLoss.item():.4f}')
        totalLoss = objectpresenceLoss + 2*positionLoss + 8*classLoss

        return totalLoss


