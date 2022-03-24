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
        self.do1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.do2 = nn.Dropout(p=0.2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.Dropout(p=0.2),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn1(self.do1(self.conv1(x))))
        #out = self.bn2(self.conv2(out))
        out = self.bn2(self.do2(self.conv2(out)))
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
        self.do1 = nn.Dropout(p=0.2)
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
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn1(self.do1(self.conv1(x))))
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
optimizer = optim.Adam(model.parameters(), lr=0.001)
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

    
   

model_path = '/scratch/ph2200/pytorch-example/project1_model_10.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4,368,690
Epoch: 0 -- Phase:train -- Loss 1.414220632095337  -- Acc tensor(47.8360, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.2052882822036743  -- Acc tensor(59.0600, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.9417403583335876  -- Acc tensor(66.2920, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 1.1360071378707886  -- Acc tensor(66.1400, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.7511904306411743  -- Acc tensor(73.3440, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.7812057035446167  -- Acc tensor(74.6800, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.6351204630470276  -- Acc tensor(77.8420, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.7670537406921387  -- Acc tensor(76.4000, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.5584577578353882  -- Acc tensor(80.5560, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.6622124296188354  -- Acc tensor(79.0700, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.4947061855316162  -- Acc tensor(82.7620, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.6453295133590699  -- Acc tensor(79.6800, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.4442709342241287  -- Acc tensor(84.3180, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.555842911529541  -- Acc tensor(82.3500, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.40579109773635863  -- Acc tensor(85.6780, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.6128649204254151  -- Acc tensor(81.8400, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.3667714319038391  -- Acc tensor(86.9920, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.5668323903083802  -- Acc tensor(83.0500, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.33207154938697814  -- Acc tensor(88.3120, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.5613423855304718  -- Acc tensor(83.5700, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.30170647795677186  -- Acc tensor(89.5020, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.551305758857727  -- Acc tensor(84.5200, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.2720725565624237  -- Acc tensor(90.2680, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.5753016240119934  -- Acc tensor(83.9100, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.24877217391967774  -- Acc tensor(91.2460, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.5165014504909515  -- Acc tensor(85.9300, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.2238944330072403  -- Acc tensor(91.9900, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.5725856544971466  -- Acc tensor(84.8200, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.20065969804763795  -- Acc tensor(92.9100, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.6233068067550659  -- Acc tensor(84.8000, device='cuda:0')
"""
