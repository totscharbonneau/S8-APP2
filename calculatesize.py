import torch
import torch.nn as nn



input_size = [1,53,53]

layer = nn.Conv2d(1,4,3,1,1)
relu1 = nn.ReLU()
maxpool1 = nn.MaxPool2d(2, 2, 0)
conv2 = nn.Conv2d(4, 16, 3, 1, 1)
relu2 = nn.ReLU()
maxpool2 = nn.MaxPool2d(2, 2, 0)
conv3 = nn.Conv2d(16, 16, 3, 1, 1)
relu3 = nn.ReLU()
maxpool3 = nn.MaxPool2d(2, 2, 0)
flatten = nn.Flatten()

num_params = sum(p.numel() for p in layer.parameters())

x = torch.randn(1, *input_size)

x = layer.forward(x)
x = relu1.forward(x)
x = maxpool1.forward(x)
x = conv2.forward(x)
x = relu2.forward(x)
x = maxpool2.forward(x)
x = conv3.forward(x)
x = relu3.forward(x)
x = maxpool3.forward(x)
out = flatten.forward(x)

print(f"Input shape : {x.shape}")
print(f"Output shape: {out.shape}")
print("Number of parameters:", num_params)