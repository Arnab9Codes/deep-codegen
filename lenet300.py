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

        self.weight = nn.Parameter(torch.Tensor(300, 100), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(100), requires_grad=True) 
        self.init_params = self.state_dict()
    
    def forward(self, x):
        x = x.view(x.size(0), 28 * 28)
        out = self.fc1(x) 
        out = self.relu1(out)
        #print(out.shape)
        out = linear_new(out, self.weight, self.bias)
        out = self.relu2(out)
        logits = self.fc3(out)
    
        return logits 


device = torch.device("cuda")
lenet_model = lenet_300().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet_model.parameters(), lr=0.001)

num_epochs = 2

for epoch in range(num_epochs):

    lenet_model.train() 
    loss_total=0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() 
        outputs = lenet_model(inputs)
        #print('outputs')
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        
        loss_total+= loss.item()
    print(loss.item())

lenet_model.eval()  
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = lenet_model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

acc = 100 * correct / total
print(f'Test Accuracy: {acc:.f}%')
