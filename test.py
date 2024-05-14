import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(33, 32, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x

    def prune(self, in_channel):
        self.conv.in_channels = self.conv.in_channels - in_channel
        self.conv.weight = nn.Parameter(self.conv.weight[:, :self.conv.in_channels, :, :])

# Create an instance of the model
model = MyModel()
model.prune(1)
test_val = torch.randn((1, 32, 32, 32), dtype = torch.float32)
test_x = torch.randn((1, 32, 32, 32), dtype = torch.float32)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
# Forward pass
output = model(test_val)
print(output)
loss = loss_func(output, test_x)
loss.backward()
optimizer.step()

print(model(test_val))

model.to('cuda')
torch.save(model.state_dict(),'test.pkl')
torch.load('test.pkl', model.state_dict())
