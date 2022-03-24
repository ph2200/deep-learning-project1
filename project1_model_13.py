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

    
  

model_path = '/scratch/ph2200/pytorch-example/project1_model_13.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4816890

Epoch: 0 -- Phase:train -- Loss 1.3769275917816162  -- Acc tensor(49.1520, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 2.6918756843566896  -- Acc tensor(26.8000, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.9659125453186035  -- Acc tensor(65.5340, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 2.3060737884521485  -- Acc tensor(30.8700, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.788197973022461  -- Acc tensor(72.0740, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 2.3932064846038816  -- Acc tensor(34.2200, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.6690811116218567  -- Acc tensor(76.5400, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 1.938087469482422  -- Acc tensor(42.5900, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.5855962950897217  -- Acc tensor(79.2760, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 1.5552294765472412  -- Acc tensor(54.4800, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.5284600487327575  -- Acc tensor(81.7300, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 1.8387442888259888  -- Acc tensor(48.2500, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.48513184779167173  -- Acc tensor(83.2240, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 1.1469092229843139  -- Acc tensor(63.4200, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.4465109475517273  -- Acc tensor(84.6060, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 1.5801656200408936  -- Acc tensor(54.2700, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.40915180574417115  -- Acc tensor(85.8520, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 1.4749265058517456  -- Acc tensor(56.5200, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.3812071859073639  -- Acc tensor(86.7640, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 1.00922174243927  -- Acc tensor(67.9400, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.3539927466964722  -- Acc tensor(87.6580, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 1.2263583806991578  -- Acc tensor(62.7900, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.3314306425762176  -- Acc tensor(88.3920, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 1.3988919956207275  -- Acc tensor(61.1500, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.3157497803401947  -- Acc tensor(88.9280, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.9940049182891846  -- Acc tensor(69.7600, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.2907209696483612  -- Acc tensor(89.8480, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.7785967050552368  -- Acc tensor(75.6400, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.277081279296875  -- Acc tensor(90.2100, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.707967206478119  -- Acc tensor(78.2000, device='cuda:0')
Epoch: 15 -- Phase:train -- Loss 0.2557749409008026  -- Acc tensor(90.9900, device='cuda:0')
Epoch: 15 -- Phase:val -- Loss 0.7864252429008484  -- Acc tensor(75.7500, device='cuda:0')
Epoch: 16 -- Phase:train -- Loss 0.24622790881633758  -- Acc tensor(91.3640, device='cuda:0')
Epoch: 16 -- Phase:val -- Loss 0.7230936876296997  -- Acc tensor(77.4300, device='cuda:0')
Epoch: 17 -- Phase:train -- Loss 0.23156890703678132  -- Acc tensor(91.7520, device='cuda:0')
Epoch: 17 -- Phase:val -- Loss 1.0108079284667968  -- Acc tensor(72.8400, device='cuda:0')
Epoch: 18 -- Phase:train -- Loss 0.21939193890094758  -- Acc tensor(92.2460, device='cuda:0')
Epoch: 18 -- Phase:val -- Loss 0.851117339515686  -- Acc tensor(76.0300, device='cuda:0')
Epoch: 19 -- Phase:train -- Loss 0.20500147916555406  -- Acc tensor(92.7060, device='cuda:0')
Epoch: 19 -- Phase:val -- Loss 0.8450517661094665  -- Acc tensor(76.3100, device='cuda:0')
Epoch: 20 -- Phase:train -- Loss 0.19304305066108704  -- Acc tensor(93.1040, device='cuda:0')
Epoch: 20 -- Phase:val -- Loss 0.7925503036499023  -- Acc tensor(77.1500, device='cuda:0')
Epoch: 21 -- Phase:train -- Loss 0.1844809433221817  -- Acc tensor(93.5220, device='cuda:0')
Epoch: 21 -- Phase:val -- Loss 0.8123611453056335  -- Acc tensor(77.3100, device='cuda:0')
Epoch: 22 -- Phase:train -- Loss 0.17393351932883264  -- Acc tensor(93.8820, device='cuda:0')
Epoch: 22 -- Phase:val -- Loss 0.640882572555542  -- Acc tensor(81.2400, device='cuda:0')
Epoch: 23 -- Phase:train -- Loss 0.16614662856698037  -- Acc tensor(94.0560, device='cuda:0')
Epoch: 23 -- Phase:val -- Loss 0.846808079624176  -- Acc tensor(78.3100, device='cuda:0')
Epoch: 24 -- Phase:train -- Loss 0.15811453717947005  -- Acc tensor(94.3420, device='cuda:0')
Epoch: 24 -- Phase:val -- Loss 0.7692006946563721  -- Acc tensor(78.8000, device='cuda:0')
Epoch: 25 -- Phase:train -- Loss 0.1536215735721588  -- Acc tensor(94.5360, device='cuda:0')
Epoch: 25 -- Phase:val -- Loss 0.6833915762901306  -- Acc tensor(81.0300, device='cuda:0')
Epoch: 26 -- Phase:train -- Loss 0.14517636734962464  -- Acc tensor(94.7060, device='cuda:0')
Epoch: 26 -- Phase:val -- Loss 0.7968075826644897  -- Acc tensor(78.9100, device='cuda:0')
Epoch: 27 -- Phase:train -- Loss 0.13754631435632705  -- Acc tensor(95.1600, device='cuda:0')
Epoch: 27 -- Phase:val -- Loss 0.6925838549613953  -- Acc tensor(81.6600, device='cuda:0')
Epoch: 28 -- Phase:train -- Loss 0.1324839842247963  -- Acc tensor(95.2200, device='cuda:0')
Epoch: 28 -- Phase:val -- Loss 0.7251210987091065  -- Acc tensor(81.0400, device='cuda:0')
Epoch: 29 -- Phase:train -- Loss 0.12746266649514437  -- Acc tensor(95.5100, device='cuda:0')
Epoch: 29 -- Phase:val -- Loss 0.7204996716499329  -- Acc tensor(80.4500, device='cuda:0')
Epoch: 30 -- Phase:train -- Loss 0.11851736543655396  -- Acc tensor(95.7560, device='cuda:0')
Epoch: 30 -- Phase:val -- Loss 0.6233126638412475  -- Acc tensor(83.0800, device='cuda:0')
Epoch: 31 -- Phase:train -- Loss 0.11669860732674599  -- Acc tensor(95.7940, device='cuda:0')
Epoch: 31 -- Phase:val -- Loss 0.6299871913909912  -- Acc tensor(83.5300, device='cuda:0')
Epoch: 32 -- Phase:train -- Loss 0.11463694009304047  -- Acc tensor(95.9140, device='cuda:0')
Epoch: 32 -- Phase:val -- Loss 0.7149612594604492  -- Acc tensor(81.2000, device='cuda:0')
Epoch: 33 -- Phase:train -- Loss 0.10168741993308067  -- Acc tensor(96.4040, device='cuda:0')
Epoch: 33 -- Phase:val -- Loss 0.7862057384490967  -- Acc tensor(81.1100, device='cuda:0')
Epoch: 34 -- Phase:train -- Loss 0.10289874813556671  -- Acc tensor(96.3120, device='cuda:0')
Epoch: 34 -- Phase:val -- Loss 0.6712199359893799  -- Acc tensor(83.4200, device='cuda:0')
Epoch: 35 -- Phase:train -- Loss 0.1007895844978094  -- Acc tensor(96.4600, device='cuda:0')
Epoch: 35 -- Phase:val -- Loss 0.6971071374893189  -- Acc tensor(82.4800, device='cuda:0')
Epoch: 36 -- Phase:train -- Loss 0.09813262700676918  -- Acc tensor(96.4400, device='cuda:0')
Epoch: 36 -- Phase:val -- Loss 0.6702241962432861  -- Acc tensor(82.7600, device='cuda:0')
Epoch: 37 -- Phase:train -- Loss 0.0923273939615488  -- Acc tensor(96.7140, device='cuda:0')
Epoch: 37 -- Phase:val -- Loss 0.6031301184654236  -- Acc tensor(84.2700, device='cuda:0')
Epoch: 38 -- Phase:train -- Loss 0.0902783465898037  -- Acc tensor(96.7420, device='cuda:0')
Epoch: 38 -- Phase:val -- Loss 0.7531862787246704  -- Acc tensor(81.8100, device='cuda:0')
Epoch: 39 -- Phase:train -- Loss 0.08649765136957169  -- Acc tensor(96.9680, device='cuda:0')
Epoch: 39 -- Phase:val -- Loss 0.8069502413749695  -- Acc tensor(81.3300, device='cuda:0')
"""