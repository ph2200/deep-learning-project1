import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
#import matplotlib.pyplot as plt
#%matplotlib inline


#load data
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=1) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=1)
# ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        #self.in_planes = 64
        self.in_planes = 40
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 40, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(40)
        #self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer1 = self._make_layer(block, 40, num_blocks[0], stride=1)
        #self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer2 = self._make_layer(block, 80, num_blocks[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer3 = self._make_layer(block, 160, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer4 = self._make_layer(block, 320, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512, num_classes)
        self.linear = nn.Linear(320, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
    return ResNet(BasicBlock, [2, 2, 2, 2])

#build model
model=project1_model().cuda()
#count parameters number
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # torch.numel() returns number of elements in a tensor

print(count_parameters(model))


# train data
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
save_loss = {'train':[], 'val':[]}
save_acc = {'train':[], 'val':[]}


for epoch in range(15):

    # Each epoch has a training and validation phase
    model.train()
    current_loss = 0.0
    current_corrects = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # print(batch_idx)
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        current_loss += loss.item() * inputs.size(0)
        current_corrects += torch.sum(preds == labels.data)

    save_loss['train'] += [current_loss / len(trainloader.dataset)]
    save_acc['train'] += [current_corrects.float() / len(trainloader.dataset)]
    # pretty print
    print("Epoch:",epoch, "-- Phase:train -- Loss",save_loss['train'][-1]," -- Acc",save_acc['train'][-1]*100)

    model.eval()
    current_loss = 0.0
    current_corrects = 0
    for batch_idx, (inputs, labels) in enumerate(testloader):
        # print(batch_idx)
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        current_loss += loss.item() * inputs.size(0)
        current_corrects += torch.sum(preds == labels.data)

    save_loss['val'] += [current_loss / len(testloader.dataset)]
    save_acc['val'] += [current_corrects.float() / len(testloader.dataset)]
    # pretty print
    #print(f"Epoch:{epoch} -- Phase:{'val'} -- Loss:{save_loss['val'][-1]:.2f} -- Acc:{save_acc['val'][-1]*100:.2f}")
    print("Epoch:",epoch, "-- Phase:val -- Loss",save_loss['val'][-1]," -- Acc",save_acc['val'][-1]*100)

    
   

model_path = '/scratch/ph2200/pytorch-example/project1_model_6.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4,368,690
Epoch: 0 -- Phase:train -- Loss 1.3040868954849243  -- Acc tensor(52.6960, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.1040408860206603  -- Acc tensor(60.7500, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.8162036298370361  -- Acc tensor(71.2060, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.7711516101837158  -- Acc tensor(72.8600, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.627914448890686  -- Acc tensor(78.0800, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.7485959115982056  -- Acc tensor(75.1500, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.514117354221344  -- Acc tensor(82.1280, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.6306595748901367  -- Acc tensor(78.5000, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.43468171829223634  -- Acc tensor(84.9800, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.5556468332290649  -- Acc tensor(81.3400, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.3635616040229797  -- Acc tensor(87.4880, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.6720009809494019  -- Acc tensor(78., device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.3055121137952805  -- Acc tensor(89.1560, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.5465984382629394  -- Acc tensor(82.5300, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.25396487932205203  -- Acc tensor(91.2120, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.5800484914302826  -- Acc tensor(81.7200, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.20751142773628234  -- Acc tensor(92.7420, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.6182847670555115  -- Acc tensor(81.0200, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.17759462246417998  -- Acc tensor(93.7320, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.614098181629181  -- Acc tensor(82.0500, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.14952752665042876  -- Acc tensor(94.7440, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.654180575466156  -- Acc tensor(81.3700, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.13457752428770065  -- Acc tensor(95.1880, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.6407886573791504  -- Acc tensor(82.4200, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.11924484028697013  -- Acc tensor(95.8440, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.6205858640670776  -- Acc tensor(82.5200, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.109395199457407  -- Acc tensor(96.2900, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.6996397390365601  -- Acc tensor(80.6400, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.10880625251173973  -- Acc tensor(96.2060, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.7210857760429382  -- Acc tensor(81.4300, device='cuda:0')
"""
