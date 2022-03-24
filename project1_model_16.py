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
optimizer =optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
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

    
  

model_path = '/scratch/ph2200/pytorch-example/project1_model_16.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4368690
Epoch: 0 -- Phase:train -- Loss 1.4433995338058472  -- Acc tensor(47.0160, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.1617370889663696  -- Acc tensor(57.9700, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.9649169273471833  -- Acc tensor(65.4760, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.892380756187439  -- Acc tensor(68.0400, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.7413575860404968  -- Acc tensor(73.7540, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.8412856609344482  -- Acc tensor(70.8800, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.581322216758728  -- Acc tensor(79.7040, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.7954479454994202  -- Acc tensor(72.6500, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.44040292514801027  -- Acc tensor(84.8060, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.775311815738678  -- Acc tensor(74.1100, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.31440397673606874  -- Acc tensor(89.2520, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.8336849534034729  -- Acc tensor(73.4900, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.20863449493408204  -- Acc tensor(93.1120, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.8908442518234253  -- Acc tensor(73.3600, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.14196484081745148  -- Acc tensor(95.2500, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.9202271230697632  -- Acc tensor(73.9900, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.09337379058122634  -- Acc tensor(97.1480, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.9763420099258423  -- Acc tensor(74.4400, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.06262841575860977  -- Acc tensor(98.1040, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.9772745464324951  -- Acc tensor(75.7100, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.04400711044937372  -- Acc tensor(98.7660, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 1.0739347988128662  -- Acc tensor(74.0700, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.03174246142297983  -- Acc tensor(99.1840, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.9852209285736084  -- Acc tensor(76.4800, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.019234864668175578  -- Acc tensor(99.5460, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.9193363063812255  -- Acc tensor(77.9800, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.011530295515507459  -- Acc tensor(99.7920, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.9002857511520386  -- Acc tensor(78.2000, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.0069466601171903316  -- Acc tensor(99.9000, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.9090340419769287  -- Acc tensor(78.5400, device='cuda:0')"""