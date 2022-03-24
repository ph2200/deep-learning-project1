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
        #self.bn1 = nn.BatchNorm2d(planes)
        self.do1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.do2 = nn.Dropout(p=0.2)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(planes)
                nn.Dropout(p=0.2)
            )

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.do1(self.conv1(x)))
        #out = self.bn2(self.conv2(out))
        out = self.do2(self.conv2(out))
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
        out = F.relu(self.do1(self.conv1(x)))
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

    
   

model_path = '/scratch/ph2200/pytorch-example/project1_model_9.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4,362,690
Epoch: 0 -- Phase:train -- Loss 1.7608182024383545  -- Acc tensor(33.5800, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.4312873237609862  -- Acc tensor(48.8400, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 1.304352099685669  -- Acc tensor(52.7200, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 1.1205383377075195  -- Acc tensor(59.8800, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 1.0875172441482543  -- Acc tensor(61.5920, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.984194197845459  -- Acc tensor(65.8200, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.9634738504791259  -- Acc tensor(65.9540, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.9092028650283813  -- Acc tensor(69.6600, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.8658664437484741  -- Acc tensor(69.7800, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.8388167505264282  -- Acc tensor(71.0200, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.795770099067688  -- Acc tensor(72.2480, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.7714952179908753  -- Acc tensor(74.0600, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.7372658039093017  -- Acc tensor(74.0760, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.7919921933174133  -- Acc tensor(73.8000, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.6999306127929688  -- Acc tensor(75.6360, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.7520934762954712  -- Acc tensor(75.1100, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.6657751567459106  -- Acc tensor(76.9280, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.6914907838821411  -- Acc tensor(76.4000, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.6340667120170593  -- Acc tensor(78.0500, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.6829697503089904  -- Acc tensor(77.1900, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.6046595612049103  -- Acc tensor(78.9360, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.6615665123939514  -- Acc tensor(77.8600, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.5861642232131958  -- Acc tensor(79.6360, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.6502222096443177  -- Acc tensor(78.3700, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.5651205735397339  -- Acc tensor(80.2560, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.6353363632202148  -- Acc tensor(78.4100, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.5474990968513489  -- Acc tensor(80.9520, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.6427527122497558  -- Acc tensor(78.2700, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.5297566856765747  -- Acc tensor(81.4740, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.6262776729106903  -- Acc tensor(79.0200, device='cuda:0')
"""
