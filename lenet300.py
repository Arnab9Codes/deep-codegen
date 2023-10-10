import torch
from torch import nn
from pytorch_apis import LinearNew

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
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root=root, train=True,transform=transform, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=transform, download=True)

batch_size = 50
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)


class lenet_300(nn.Module):
    def __init__(self):
        super(lenet_300, self).__init__()
        self.device =torch.device('cuda')
        self.ln1 = LinearNew(784, 300, self.device)
        self.relu1 = nn.ReLU()
        self.ln2 = LinearNew(300, 100, self.device)
        self.relu2 = nn.ReLU()
        self.ln3 = LinearNew(100, 10, self.device)
        self.init_params = self.state_dict()
    
    def forward(self, x):
        x = x.view(x.size(0), 28 * 28)
        out = self.ln1(x)
        out = self.relu1(out)
        out = self.ln2(out)
        out = self.relu2(out)
        logits = self.ln3(out)
    
        return logits 


device = torch.device("cuda")
lenet_model = lenet_300().to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
#optimizer = optim.SGD(lenet_model.parameters(), lr=0.001) 
optimizer = optim.Adam(lenet_model.parameters(), lr=learning_rate)

num_epochs = 25

for epoch in range(num_epochs):

    lenet_model.train() 
    loss_total=0.0
    loss =0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() 
        outputs = lenet_model(inputs)
        #print('outputs')
        loss = criterion(outputs, targets)
        #print(loss.item())

        loss.backward()
        optimizer.step()
        
        #loss_total+= loss.item()
    print('epoch: ', epoch, 'loss: ',loss.item())

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
print(f'Test Accuracy: {acc}%')

torch.save(lenet_300, './state_dict/lenet_traned.pth')
