import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
#import matplotlib.pyplot as plt
#%matplotlib inline


#load data
transform_train = torchvision.transforms.Compose([
                    #torchvision.transforms.RandomCrop(32, padding=4),
                    #torchvision.transforms.RandomHorizontalFlip(),
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
#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer =optim.SGD(model.parameters(), lr=0.001,momentum=0.99)
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

    
  

model_path = '/scratch/ph2200/pytorch-example/project1_model_15.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4368690
Epoch: 0 -- Phase:train -- Loss 1.4668026622390746  -- Acc tensor(46.1580, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.0501414545059204  -- Acc tensor(62.2400, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.9513446823310852  -- Acc tensor(66.2180, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.8794527799606323  -- Acc tensor(68.9200, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.6886109619808197  -- Acc tensor(75.5940, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.6974734153747558  -- Acc tensor(75.4500, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.5356731602478028  -- Acc tensor(81.2080, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.667827218914032  -- Acc tensor(77.3000, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.4120091365623474  -- Acc tensor(85.4920, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.6507365065097809  -- Acc tensor(78.7200, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.32593444568634033  -- Acc tensor(88.5460, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.6768436576843262  -- Acc tensor(78.7300, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.22874690112352372  -- Acc tensor(92.1060, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.6775656457901001  -- Acc tensor(79.8800, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.16552929248332976  -- Acc tensor(94.1820, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.7010985843658447  -- Acc tensor(79.9600, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.14063906393051148  -- Acc tensor(94.8440, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.7122112929344178  -- Acc tensor(81.2300, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.09390244495153427  -- Acc tensor(96.7400, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.8179071749687195  -- Acc tensor(79.8800, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.06094635268092156  -- Acc tensor(97.8700, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.8637169659614563  -- Acc tensor(81.4400, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.05977238630414009  -- Acc tensor(97.9100, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.8994339656829834  -- Acc tensor(80.3300, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.04937757630213164  -- Acc tensor(98.3160, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.869471534729004  -- Acc tensor(81.0500, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.03784432712242007  -- Acc tensor(98.7260, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.9119036310195923  -- Acc tensor(81.2200, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.031238877731580288  -- Acc tensor(98.9780, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.8983924468994141  -- Acc tensor(81.8900, device='cuda:0')"""