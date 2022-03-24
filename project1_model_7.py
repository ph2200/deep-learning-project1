import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
#import matplotlib.pyplot as plt
#%matplotlib inline


#load data
transform_train = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=1) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
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
        self.layer1 = self._make_layer(block, 42, num_blocks[0], stride=1)
        #self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer2 = self._make_layer(block, 84, num_blocks[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer3 = self._make_layer(block, 168, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer4 = self._make_layer(block, 336, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512, num_classes)
        self.linear = nn.Linear(336, num_classes)

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
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.001)
save_loss = {'train':[], 'val':[]}
save_acc = {'train':[], 'val':[]}


for epoch in range(40):

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

    
  

model_path = '/scratch/ph2200/pytorch-example/project1_model_7.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4816890

Epoch: 0 -- Phase:train -- Loss 1.3822812524795531  -- Acc tensor(49.2260, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.266452688598633  -- Acc tensor(54.9600, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 1.018773097076416  -- Acc tensor(63.6780, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.9796818904876708  -- Acc tensor(66.1000, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.8692299831390381  -- Acc tensor(69.9400, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 1.0224953451156615  -- Acc tensor(65.0500, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.7581578990554809  -- Acc tensor(74.0400, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 1.0682902660369873  -- Acc tensor(66.8900, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.6778573930549622  -- Acc tensor(76.7460, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.7132135088920594  -- Acc tensor(75.8400, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.6309935593414306  -- Acc tensor(78.4280, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.6587300206184388  -- Acc tensor(77.0900, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.5891264253616333  -- Acc tensor(79.9160, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.6013139595031738  -- Acc tensor(79.5000, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.5626671852588654  -- Acc tensor(80.6500, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.7822243930339813  -- Acc tensor(74.3000, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.5389220907211304  -- Acc tensor(81.5740, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.6086796796798706  -- Acc tensor(80.2300, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.522294183959961  -- Acc tensor(82.2780, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.5802430401325226  -- Acc tensor(80.6100, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.5023720764160157  -- Acc tensor(82.8740, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.5706584057807922  -- Acc tensor(80.1800, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.48602394523620607  -- Acc tensor(83.2720, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.5053914538383484  -- Acc tensor(82.7200, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.4735792647266388  -- Acc tensor(83.8660, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.5218922894477844  -- Acc tensor(82.4200, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.46876405292510986  -- Acc tensor(84.0820, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.4429271161556244  -- Acc tensor(84.9700, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.45790921012878416  -- Acc tensor(84.4580, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.49421535959243773  -- Acc tensor(83.6400, device='cuda:0')
Epoch: 15 -- Phase:train -- Loss 0.4507238701248169  -- Acc tensor(84.6820, device='cuda:0')
Epoch: 15 -- Phase:val -- Loss 0.5387166570186614  -- Acc tensor(81.9300, device='cuda:0')
Epoch: 16 -- Phase:train -- Loss 0.44711829763412475  -- Acc tensor(84.7840, device='cuda:0')
Epoch: 16 -- Phase:val -- Loss 0.46812553558349607  -- Acc tensor(84.1100, device='cuda:0')
Epoch: 17 -- Phase:train -- Loss 0.44336033206939696  -- Acc tensor(85.0540, device='cuda:0')
Epoch: 17 -- Phase:val -- Loss 0.43127860450744626  -- Acc tensor(85.1600, device='cuda:0')
Epoch: 18 -- Phase:train -- Loss 0.432019598608017  -- Acc tensor(85.4120, device='cuda:0')
Epoch: 18 -- Phase:val -- Loss 0.48357577548027036  -- Acc tensor(83.6500, device='cuda:0')
Epoch: 19 -- Phase:train -- Loss 0.4279165819549561  -- Acc tensor(85.5480, device='cuda:0')
Epoch: 19 -- Phase:val -- Loss 0.46243967690467835  -- Acc tensor(84.4400, device='cuda:0')
Epoch: 20 -- Phase:train -- Loss 0.42211009086608886  -- Acc tensor(85.8080, device='cuda:0')
Epoch: 20 -- Phase:val -- Loss 0.516564953994751  -- Acc tensor(82.7000, device='cuda:0')
Epoch: 21 -- Phase:train -- Loss 0.4154148382902145  -- Acc tensor(85.9180, device='cuda:0')
Epoch: 21 -- Phase:val -- Loss 0.4984132761001587  -- Acc tensor(83.5000, device='cuda:0')
Epoch: 22 -- Phase:train -- Loss 0.4171704113769531  -- Acc tensor(85.8820, device='cuda:0')
Epoch: 22 -- Phase:val -- Loss 0.4760319371700287  -- Acc tensor(83.7400, device='cuda:0')
Epoch: 23 -- Phase:train -- Loss 0.41009297223091123  -- Acc tensor(86.0000, device='cuda:0')
Epoch: 23 -- Phase:val -- Loss 0.4570924695968628  -- Acc tensor(84.5600, device='cuda:0')
Epoch: 24 -- Phase:train -- Loss 0.4108639611721039  -- Acc tensor(86.0260, device='cuda:0')
Epoch: 24 -- Phase:val -- Loss 0.5434071271419525  -- Acc tensor(82.0800, device='cuda:0')
Epoch: 25 -- Phase:train -- Loss 0.40755724477767946  -- Acc tensor(86.1020, device='cuda:0')
Epoch: 25 -- Phase:val -- Loss 0.42347150468826295  -- Acc tensor(85.7600, device='cuda:0')
Epoch: 26 -- Phase:train -- Loss 0.40545944454193117  -- Acc tensor(86.1400, device='cuda:0')
Epoch: 26 -- Phase:val -- Loss 0.6195221700668335  -- Acc tensor(81.0300, device='cuda:0')
Epoch: 27 -- Phase:train -- Loss 0.3970625202560425  -- Acc tensor(86.4660, device='cuda:0')
Epoch: 27 -- Phase:val -- Loss 0.5590090463638305  -- Acc tensor(81.8700, device='cuda:0')
Epoch: 28 -- Phase:train -- Loss 0.39432354734420777  -- Acc tensor(86.5280, device='cuda:0')
Epoch: 28 -- Phase:val -- Loss 0.47891723675727843  -- Acc tensor(84.0200, device='cuda:0')
Epoch: 29 -- Phase:train -- Loss 0.39682589162826537  -- Acc tensor(86.4120, device='cuda:0')
Epoch: 29 -- Phase:val -- Loss 0.5346150283813477  -- Acc tensor(83.3500, device='cuda:0')
Epoch: 30 -- Phase:train -- Loss 0.3920129232788086  -- Acc tensor(86.7040, device='cuda:0')
Epoch: 30 -- Phase:val -- Loss 0.4641786808013916  -- Acc tensor(84.5200, device='cuda:0')
Epoch: 31 -- Phase:train -- Loss 0.38795764945983885  -- Acc tensor(86.8860, device='cuda:0')
Epoch: 31 -- Phase:val -- Loss 0.45984208030700685  -- Acc tensor(84.7000, device='cuda:0')
Epoch: 32 -- Phase:train -- Loss 0.38950614721298216  -- Acc tensor(86.7500, device='cuda:0')
Epoch: 32 -- Phase:val -- Loss 0.42386397709846496  -- Acc tensor(85.9900, device='cuda:0')
Epoch: 33 -- Phase:train -- Loss 0.3854212857341766  -- Acc tensor(86.9940, device='cuda:0')
Epoch: 33 -- Phase:val -- Loss 0.39564253153800966  -- Acc tensor(86.5700, device='cuda:0')
Epoch: 34 -- Phase:train -- Loss 0.38566339854240417  -- Acc tensor(86.9560, device='cuda:0')
Epoch: 34 -- Phase:val -- Loss 0.41706049671173095  -- Acc tensor(85.9500, device='cuda:0')
Epoch: 35 -- Phase:train -- Loss 0.38173051705121996  -- Acc tensor(86.9860, device='cuda:0')
Epoch: 35 -- Phase:val -- Loss 0.45293931455612185  -- Acc tensor(85.0600, device='cuda:0')
Epoch: 36 -- Phase:train -- Loss 0.38014261491298673  -- Acc tensor(87.1100, device='cuda:0')
Epoch: 36 -- Phase:val -- Loss 0.46089492762088774  -- Acc tensor(84.9900, device='cuda:0')
Epoch: 37 -- Phase:train -- Loss 0.38423521743774414  -- Acc tensor(86.9360, device='cuda:0')
Epoch: 37 -- Phase:val -- Loss 0.4261561348438263  -- Acc tensor(85.2700, device='cuda:0')
Epoch: 38 -- Phase:train -- Loss 0.3813898586750031  -- Acc tensor(86.8060, device='cuda:0')
Epoch: 38 -- Phase:val -- Loss 0.4189528985500336  -- Acc tensor(85.8500, device='cuda:0')
Epoch: 39 -- Phase:train -- Loss 0.37379716876983643  -- Acc tensor(87.3400, device='cuda:0')
Epoch: 39 -- Phase:val -- Loss 0.47101856384277346  -- Acc tensor(84.2600, device='cuda:0')
"""