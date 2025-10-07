import torch.nn as nn

class Class_nn(nn.Module):
    def __init__(self):
        super(Class_nn, self).__init__()

        self.conv1 = nn.Conv2d(1,16,5,1,1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2,0)
        self.conv2 = nn.Conv2d(16,32,5,1,1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2,0)
        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, 1, 0)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(6400, 16)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(16,3)


    def forward(self,x):
        x = self.conv1.forward(x) #16x51x51
        x = self.relu1.forward(x) #16x51x51
        x = self.maxpool1.forward(x) #16x25x25
        x = self.conv2.forward(x) #32x23x23
        x = self.relu2.forward(x) #32x23x23
        x = self.maxpool2.forward(x) #32x11x11
        x = self.conv3.forward(x) #64x11x11
        x = self.relu3.forward(x) #64x11x11
        x = self.maxpool3.forward(x) #64x10x10

        x = self.flatten.forward(x) #6400
        x = self.fc1.forward(x) #16
        x = self.relu5.forward(x) #16
        x = self.fc2.forward(x) #3
        # output = self.sigma.forward(x)
        return x


    def count_parameters(self):
        parameters = 0
        for name, layer in self.named_children():
            currenttotal = sum((p.numel() for p in layer.parameters()))
            parameters += currenttotal
            if (currenttotal > 0):
                print(f"{name} : {currenttotal} parameters")

        print(f"Total parameters : {parameters}")

    def compute_receptive_field(self, input_size=53, verbose=True):
        rf = 1  # receptive field
        jump = 1  # effective stride
        size = input_size
        results = []

        def layer_info(layer, rf, jump, size):
            if isinstance(layer, nn.Conv2d):
                k = layer.kernel_size[0]
                s = layer.stride[0]
                p = layer.padding[0]
                rf_new = rf + (k - 1) * jump
                jump_new = jump * s
                size_new = (size + 2 * p - k) // s + 1
            elif isinstance(layer, nn.MaxPool2d):
                k = layer.kernel_size
                s = layer.stride
                p = layer.padding
                rf_new = rf + (k - 1) * jump
                jump_new = jump * s
                size_new = (size + 2 * p - k) // s + 1
            else:
                return rf, jump, size
            return rf_new, jump_new, size_new

        for name, layer in self.named_children():
            old_rf, old_jump, old_size = rf, jump, size
            rf, jump, size = layer_info(layer, rf, jump, size)
            if rf != old_rf or jump != old_jump:
                results.append({
                    'layer': name,
                    'receptive_field': rf,
                    'jump': jump,
                    'feature_map_size': size
                })

        if verbose:
            for r in results:
                print(f"{r['layer']:10s} | RF={r['receptive_field']:3d} | "
                      f"jump={r['jump']:2d} | map={r['feature_map_size']:2d}")

            print(f"\nFinal receptive field: {results[-1]['receptive_field']} pixels")

        return results


# test = Class_nn()
#
# test.compute_receptive_field()