import torch
from torch import nn
from pytorch_apis import linear_new

import numpy as np
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch.optim as optim

root = './mnist'
if not os.path.exists(root):
    os.mkdir(root)

transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root=root, train=True,transform=transform, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=transform, download=True)

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)



class lenet_300(nn.Module):
    def __init__(self):
        super(lenet_300, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.relu1 = nn.ReLU(inplace = True)
    
        self.relu2 = nn.ReLU(inplace = True)
    
        self.fc3 = nn.Linear(100, 10)
    
        self.init_params = self.state_dict()

        self.weight = nn.Parameter(torch.Tensor(300, 100))
        self.bias = nn.Parameter(torch.Tensor(100)) 
    
    def forward(self, x):
        x = x.view(x.size(0), 28 * 28)
        out = self.fc1(x)
        out = self.relu1(out)
        out = linear_new(out, self.weight, self.bias)
        out = self.relu2(out)
        logits = nn.Softmax(self.fc3(out), dim = 0)
    
        return logits 

lenet_model = lenet_300()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet_model.parameters(), lr=0.001)

num_epochs = 2

for epoch in range(num_epochs):

    lenet_model.train() 
    loss_total=0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        

        optimizer.zero_grad() 
        outputs = lenet_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        loss_total+= loss.item()
