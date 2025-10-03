import torch.nn as nn

class Class_nn(nn.Module):
    def __init__(self):
        super(Class_nn, self).__init__()

        self.conv1 = nn.Conv2d(1,4,3,1,1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2,0)
        self.conv2 = nn.Conv2d(4,16,3,1,1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2,0)
        self.conv3 = nn.Conv2d(16,16,3,1,1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, 2, 0)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(576, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256,3)
        self.sigma = nn.Sigmoid()


    def forward(self,x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.maxpool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.maxpool2.forward(x)
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.maxpool3.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu4.forward(x)
        x = self.fc2.forward(x)
        output = self.sigma.forward(x)
        return output


    def count_parameters(self):
        parameters = 0
        for name, layer in self.named_children():
            currenttotal = sum((p.numel() for p in layer.parameters()))
            parameters += currenttotal
            if (currenttotal > 0):
                print(f"{name} : {currenttotal} parameters")

        print(f"Total parameters : {parameters}")
