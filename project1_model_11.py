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
        out = F.relu(self.bn1(self.bn1(self.conv1(x))))
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

    
  

model_path = '/scratch/ph2200/pytorch-example/project1_model_11.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4816890

Epoch: 0 -- Phase:train -- Loss 1.495862273979187  -- Acc tensor(44.2880, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 3.4915042514801025  -- Acc tensor(21.5600, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 1.1091211415100097  -- Acc tensor(59.9860, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 2.931534825515747  -- Acc tensor(30.0100, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.9384714211273193  -- Acc tensor(66.5760, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 2.010341377258301  -- Acc tensor(46.1700, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.8170374536514282  -- Acc tensor(71.0360, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 2.0379695861816405  -- Acc tensor(46.4400, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.7171539492034912  -- Acc tensor(74.9200, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 1.7004369440078735  -- Acc tensor(53.1300, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.6520694521903991  -- Acc tensor(77.1860, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 1.3333980043411255  -- Acc tensor(60.4600, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.5968257650375366  -- Acc tensor(79.3080, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 1.6113320854187012  -- Acc tensor(58.2100, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.5536857823944091  -- Acc tensor(80.9980, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 1.2659904321670532  -- Acc tensor(61.9800, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.5184911796474457  -- Acc tensor(82.0260, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.9347432430267334  -- Acc tensor(72.0300, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.49183775654792783  -- Acc tensor(82.8280, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 1.026058170890808  -- Acc tensor(70.1700, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.46341711678504943  -- Acc tensor(83.9980, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 1.448563005065918  -- Acc tensor(64.7200, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.4401309450817108  -- Acc tensor(84.6940, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.9610926441192627  -- Acc tensor(71.9900, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.41910900374412535  -- Acc tensor(85.4440, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.8524701252937317  -- Acc tensor(73.9300, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.39998574828147887  -- Acc tensor(86.1920, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.9165342219352722  -- Acc tensor(73.5800, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.3864987723922729  -- Acc tensor(86.5760, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.676977190876007  -- Acc tensor(78.3900, device='cuda:0')
Epoch: 15 -- Phase:train -- Loss 0.37113559292793274  -- Acc tensor(87.2300, device='cuda:0')
Epoch: 15 -- Phase:val -- Loss 0.850819596195221  -- Acc tensor(75.5600, device='cuda:0')
Epoch: 16 -- Phase:train -- Loss 0.3593602436876297  -- Acc tensor(87.6340, device='cuda:0')
Epoch: 16 -- Phase:val -- Loss 0.8027372667312622  -- Acc tensor(77.9000, device='cuda:0')
Epoch: 17 -- Phase:train -- Loss 0.3437785091304779  -- Acc tensor(88.1480, device='cuda:0')
Epoch: 17 -- Phase:val -- Loss 0.8292715761184692  -- Acc tensor(77.3400, device='cuda:0')
Epoch: 18 -- Phase:train -- Loss 0.3316911460113525  -- Acc tensor(88.4020, device='cuda:0')
Epoch: 18 -- Phase:val -- Loss 0.7159517412185669  -- Acc tensor(79.4700, device='cuda:0')
Epoch: 19 -- Phase:train -- Loss 0.31884944917201996  -- Acc tensor(89.0240, device='cuda:0')
Epoch: 19 -- Phase:val -- Loss 0.7203234132766724  -- Acc tensor(79.4600, device='cuda:0')
Epoch: 20 -- Phase:train -- Loss 0.3072702047729492  -- Acc tensor(89.3400, device='cuda:0')
Epoch: 20 -- Phase:val -- Loss 0.6526672966957092  -- Acc tensor(80.7600, device='cuda:0')
Epoch: 21 -- Phase:train -- Loss 0.3019871421670914  -- Acc tensor(89.4800, device='cuda:0')
Epoch: 21 -- Phase:val -- Loss 0.6911936756134033  -- Acc tensor(80.0000, device='cuda:0')
Epoch: 22 -- Phase:train -- Loss 0.28996622754096985  -- Acc tensor(89.8540, device='cuda:0')
Epoch: 22 -- Phase:val -- Loss 0.6408847133636475  -- Acc tensor(80.4800, device='cuda:0')
Epoch: 23 -- Phase:train -- Loss 0.2806473952770233  -- Acc tensor(90.2760, device='cuda:0')
Epoch: 23 -- Phase:val -- Loss 0.6868017887592316  -- Acc tensor(80.6700, device='cuda:0')
Epoch: 24 -- Phase:train -- Loss 0.2751770967245102  -- Acc tensor(90.4340, device='cuda:0')
Epoch: 24 -- Phase:val -- Loss 0.7659010149002076  -- Acc tensor(78.0900, device='cuda:0')
Epoch: 25 -- Phase:train -- Loss 0.2634838057899475  -- Acc tensor(90.7860, device='cuda:0')
Epoch: 25 -- Phase:val -- Loss 0.5996730485916137  -- Acc tensor(82.4500, device='cuda:0')
Epoch: 26 -- Phase:train -- Loss 0.2607620955038071  -- Acc tensor(90.9680, device='cuda:0')
Epoch: 26 -- Phase:val -- Loss 0.6965479557991028  -- Acc tensor(80.8900, device='cuda:0')
Epoch: 27 -- Phase:train -- Loss 0.25106367447137834  -- Acc tensor(91.1840, device='cuda:0')
Epoch: 27 -- Phase:val -- Loss 0.5962323261260987  -- Acc tensor(83.2400, device='cuda:0')
Epoch: 28 -- Phase:train -- Loss 0.24720633103847503  -- Acc tensor(91.1940, device='cuda:0')
Epoch: 28 -- Phase:val -- Loss 0.5160473954200745  -- Acc tensor(84.4200, device='cuda:0')
Epoch: 29 -- Phase:train -- Loss 0.23994627906799315  -- Acc tensor(91.5040, device='cuda:0')
Epoch: 29 -- Phase:val -- Loss 0.5465379994392395  -- Acc tensor(84.4900, device='cuda:0')
Epoch: 30 -- Phase:train -- Loss 0.2334111022901535  -- Acc tensor(91.8300, device='cuda:0')
Epoch: 30 -- Phase:val -- Loss 0.7026427896022797  -- Acc tensor(81.5600, device='cuda:0')
Epoch: 31 -- Phase:train -- Loss 0.22593407849311828  -- Acc tensor(92.0240, device='cuda:0')
Epoch: 31 -- Phase:val -- Loss 0.6034777933120727  -- Acc tensor(82.6100, device='cuda:0')
Epoch: 32 -- Phase:train -- Loss 0.2206020289516449  -- Acc tensor(92.1140, device='cuda:0')
Epoch: 32 -- Phase:val -- Loss 0.5652514931678772  -- Acc tensor(83.1500, device='cuda:0')
Epoch: 33 -- Phase:train -- Loss 0.21394793438911439  -- Acc tensor(92.4900, device='cuda:0')
Epoch: 33 -- Phase:val -- Loss 0.536800904083252  -- Acc tensor(84.8600, device='cuda:0')
Epoch: 34 -- Phase:train -- Loss 0.2108361137276888  -- Acc tensor(92.7260, device='cuda:0')
Epoch: 34 -- Phase:val -- Loss 0.6020545755386353  -- Acc tensor(83.1900, device='cuda:0')
Epoch: 35 -- Phase:train -- Loss 0.20635762088298798  -- Acc tensor(92.6900, device='cuda:0')
Epoch: 35 -- Phase:val -- Loss 0.564140605545044  -- Acc tensor(84.3700, device='cuda:0')
Epoch: 36 -- Phase:train -- Loss 0.20236471092700958  -- Acc tensor(92.7560, device='cuda:0')
Epoch: 36 -- Phase:val -- Loss 0.5592765861034393  -- Acc tensor(85.3200, device='cuda:0')
Epoch: 37 -- Phase:train -- Loss 0.2000198425102234  -- Acc tensor(92.9100, device='cuda:0')
Epoch: 37 -- Phase:val -- Loss 0.5786831405639649  -- Acc tensor(84.5600, device='cuda:0')
Epoch: 38 -- Phase:train -- Loss 0.19231780615329744  -- Acc tensor(93.2120, device='cuda:0')
Epoch: 38 -- Phase:val -- Loss 0.7198079445838929  -- Acc tensor(82.6900, device='cuda:0')
Epoch: 39 -- Phase:train -- Loss 0.18810128574609755  -- Acc tensor(93.3480, device='cuda:0')
Epoch: 39 -- Phase:val -- Loss 0.6496960222244262  -- Acc tensor(84.3600, device='cuda:0')
"""
