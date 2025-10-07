import torch.nn as nn
import torch
import torch.nn.functional as F


class Det_nn(nn.Module):
    def __init__(self, input_channels=1, n_objects=3, n_outputs_per_object=7):
        super(Det_nn, self).__init__()
        self.n_obj = n_objects
        self.n_out = n_outputs_per_object
        self.conv_1 = nn.Conv2d(input_channels, 32, 5, 1, 1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(2, 2)  # (53x53) → (26x26)
        self.conv_2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(2, 2)  # (26x26) → (13x13)
        self.conv_3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.relu_3 = nn.ReLU()
        self.conv_4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.relu_4 = nn.ReLU()
        self.maxpool_4 = nn.MaxPool2d(2, 2)  # (13x13) → (6x6)

        self.flatten_5 = nn.Flatten()
        self.linear_6 = nn.Linear(64 * 6 * 6, 64)
        self.relu_6 = nn.ReLU()
        # self.linear_7 = nn.Linear(128, 64)
        # self.relu_7 = nn.ReLU()
        self.linear_8 = nn.Linear(64, n_objects * n_outputs_per_object)
        # self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x= self.conv_1.forward(x)
        x= self.bn_1.forward(x)
        x= self.relu_1.forward(x)
        x= self.maxpool_1.forward(x)
        x= self.conv_2.forward(x)
        x= self.bn_2.forward(x)
        x= self.relu_2.forward(x)
        x= self.maxpool_2.forward(x)
        x= self.conv_3.forward(x)
        x= self.bn_3.forward(x)
        x= self.relu_3.forward(x)
        x= self.conv_4.forward(x)
        x= self.bn_4.forward(x)
        x= self.relu_4.forward(x)
        x= self.maxpool_4.forward(x)
        x= self.flatten_5.forward(x)
        x= self.linear_6.forward(x)
        x= self.relu_6.forward(x)
        # x = self.dropout(x)
        # x = self.linear_7.forward(x)
        # x = self.relu_7.forward(x)
        x= self.linear_8.forward(x)
        x = x.view(-1,self.n_obj,self.n_out)
        x_out = torch.cat([
            torch.sigmoid(x[:, :, 0:4]),  # objectness, x, y, wh
            torch.sigmoid(x[:, :, 4:7])  # class1, class2, class3 (raw logits)
        ], dim=-1)
        return x_out

    def count_parameters(self):
        parameters = 0
        for name, layer in self.named_children():
            currenttotal = sum((p.numel() for p in layer.parameters()))
            parameters += currenttotal
            if currenttotal > 0:
                print(f"{name} : {currenttotal} parameters")
        print(f"Total parameters : {parameters}")


# det = Det_nn()
# det.count_parameters()
