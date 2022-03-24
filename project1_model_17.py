import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
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


  
#accuracy
plt.plot(torch.tensor(save_acc['train']))
plt.plot(torch.tensor(save_acc['val']))
plt.legend(["train", "val"])
plt.title("Accuracy")    
plt.savefig('/scratch/ph2200/pytorch-example/test_accuracy_model17')

plt.cla()
#loss
plt.plot(torch.tensor(save_loss['train']))
plt.plot(torch.tensor(save_loss['val']))
plt.legend(["train", "val"])
plt.title("Loss")    
plt.savefig('/scratch/ph2200/pytorch-example/test_loss_model17')
     
  

model_path = '/scratch/ph2200/pytorch-example/project1_model_17.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4,368,690
Epoch: 0 -- Phase:train -- Loss 1.259496775894165  -- Acc tensor(54.4300, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 0.887058869934082  -- Acc tensor(68.8200, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.7816207250976562  -- Acc tensor(72.7360, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.7120188099861146  -- Acc tensor(75.4500, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.5892700457954406  -- Acc tensor(79.4400, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.613743757724762  -- Acc tensor(78.4300, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.4682417111206055  -- Acc tensor(83.7760, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.5810870047569275  -- Acc tensor(80.4600, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.36680484043121336  -- Acc tensor(87.2040, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.5840448922157288  -- Acc tensor(80.7200, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.27986344455242157  -- Acc tensor(90.1220, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.5353043081283569  -- Acc tensor(82.5100, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.19999993937969207  -- Acc tensor(92.9560, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.5966605701446533  -- Acc tensor(81.9400, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.15228519476890565  -- Acc tensor(94.6860, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.6186674123764038  -- Acc tensor(83.3100, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.11078184492588043  -- Acc tensor(96.1480, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.6798389678955078  -- Acc tensor(82.7000, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.09844467894405126  -- Acc tensor(96.5820, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.7218766745567322  -- Acc tensor(81.9900, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.07611752858251333  -- Acc tensor(97.2980, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.8302672894477844  -- Acc tensor(81.4500, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.06678034356713294  -- Acc tensor(97.6680, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.796807626247406  -- Acc tensor(82.5000, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.06201236922115087  -- Acc tensor(97.8800, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.7673347584724426  -- Acc tensor(82.8400, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.05837172786235809  -- Acc tensor(97.9220, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.8052581562042236  -- Acc tensor(82.8800, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.046323104571700094  -- Acc tensor(98.3820, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.8833374160766602  -- Acc tensor(81.9600, device='cuda:0')
Epoch: 15 -- Phase:train -- Loss 0.046697109363526106  -- Acc tensor(98.3640, device='cuda:0')
Epoch: 15 -- Phase:val -- Loss 0.8962502370834351  -- Acc tensor(81.8700, device='cuda:0')
Epoch: 16 -- Phase:train -- Loss 0.04617232714548707  -- Acc tensor(98.3720, device='cuda:0')
Epoch: 16 -- Phase:val -- Loss 0.8246406154632568  -- Acc tensor(83.1300, device='cuda:0')
Epoch: 17 -- Phase:train -- Loss 0.03857297271691263  -- Acc tensor(98.7040, device='cuda:0')
Epoch: 17 -- Phase:val -- Loss 0.8965242866516113  -- Acc tensor(82.4000, device='cuda:0')
Epoch: 18 -- Phase:train -- Loss 0.039312521762698886  -- Acc tensor(98.5980, device='cuda:0')
Epoch: 18 -- Phase:val -- Loss 0.8874909643173218  -- Acc tensor(83.4300, device='cuda:0')
Epoch: 19 -- Phase:train -- Loss 0.03464733778908849  -- Acc tensor(98.7440, device='cuda:0')
Epoch: 19 -- Phase:val -- Loss 0.8687629131317138  -- Acc tensor(83.0500, device='cuda:0')
Epoch: 20 -- Phase:train -- Loss 0.030326256414689123  -- Acc tensor(98.9340, device='cuda:0')
Epoch: 20 -- Phase:val -- Loss 0.9191121693611145  -- Acc tensor(82.5400, device='cuda:0')
Epoch: 21 -- Phase:train -- Loss 0.03263617315173149  -- Acc tensor(98.8920, device='cuda:0')
Epoch: 21 -- Phase:val -- Loss 0.8741509177207947  -- Acc tensor(82.8700, device='cuda:0')
Epoch: 22 -- Phase:train -- Loss 0.028386960270032287  -- Acc tensor(99.0300, device='cuda:0')
Epoch: 22 -- Phase:val -- Loss 0.949011562538147  -- Acc tensor(82.7800, device='cuda:0')
Epoch: 23 -- Phase:train -- Loss 0.031027662982512267  -- Acc tensor(98.9300, device='cuda:0')
Epoch: 23 -- Phase:val -- Loss 0.9190639702796936  -- Acc tensor(83.8200, device='cuda:0')
Epoch: 24 -- Phase:train -- Loss 0.032610597347281875  -- Acc tensor(98.8380, device='cuda:0')
Epoch: 24 -- Phase:val -- Loss 0.943613641834259  -- Acc tensor(82.8800, device='cuda:0')
Epoch: 25 -- Phase:train -- Loss 0.02300875730611384  -- Acc tensor(99.1580, device='cuda:0')
Epoch: 25 -- Phase:val -- Loss 0.9080980540275574  -- Acc tensor(83.0200, device='cuda:0')
Epoch: 26 -- Phase:train -- Loss 0.027357300318554045  -- Acc tensor(99.0160, device='cuda:0')
Epoch: 26 -- Phase:val -- Loss 1.1607788009643554  -- Acc tensor(80.5100, device='cuda:0')
Epoch: 27 -- Phase:train -- Loss 0.022631040228009224  -- Acc tensor(99.2560, device='cuda:0')
Epoch: 27 -- Phase:val -- Loss 0.9252754830360412  -- Acc tensor(83.1400, device='cuda:0')
Epoch: 28 -- Phase:train -- Loss 0.0262917307563778  -- Acc tensor(99.1080, device='cuda:0')
Epoch: 28 -- Phase:val -- Loss 0.9420547973632812  -- Acc tensor(82.6800, device='cuda:0')
Epoch: 29 -- Phase:train -- Loss 0.01645536526856944  -- Acc tensor(99.4520, device='cuda:0')
Epoch: 29 -- Phase:val -- Loss 0.937023644542694  -- Acc tensor(83.8300, device='cuda:0')
Epoch: 30 -- Phase:train -- Loss 0.0198775023689121  -- Acc tensor(99.3100, device='cuda:0')
Epoch: 30 -- Phase:val -- Loss 0.9909688655853272  -- Acc tensor(83.0700, device='cuda:0')
Epoch: 31 -- Phase:train -- Loss 0.023558229414261878  -- Acc tensor(99.1840, device='cuda:0')
Epoch: 31 -- Phase:val -- Loss 0.9720621829986572  -- Acc tensor(83.4800, device='cuda:0')
Epoch: 32 -- Phase:train -- Loss 0.02041095975832548  -- Acc tensor(99.2900, device='cuda:0')
Epoch: 32 -- Phase:val -- Loss 0.9937717741012573  -- Acc tensor(83.2400, device='cuda:0')
Epoch: 33 -- Phase:train -- Loss 0.02154456074095797  -- Acc tensor(99.2680, device='cuda:0')
Epoch: 33 -- Phase:val -- Loss 1.0087039188146592  -- Acc tensor(82.9300, device='cuda:0')
Epoch: 34 -- Phase:train -- Loss 0.016705210190508513  -- Acc tensor(99.4260, device='cuda:0')
Epoch: 34 -- Phase:val -- Loss 0.9553451942443848  -- Acc tensor(83.8800, device='cuda:0')
Epoch: 35 -- Phase:train -- Loss 0.019120302129741758  -- Acc tensor(99.3540, device='cuda:0')
Epoch: 35 -- Phase:val -- Loss 1.0404140031814575  -- Acc tensor(83.0500, device='cuda:0')
Epoch: 36 -- Phase:train -- Loss 0.01876854506328702  -- Acc tensor(99.3480, device='cuda:0')
Epoch: 36 -- Phase:val -- Loss 0.9532024188995362  -- Acc tensor(83.6700, device='cuda:0')
Epoch: 37 -- Phase:train -- Loss 0.018938841955121608  -- Acc tensor(99.3280, device='cuda:0')
Epoch: 37 -- Phase:val -- Loss 0.9778423614501953  -- Acc tensor(83.3000, device='cuda:0')
Epoch: 38 -- Phase:train -- Loss 0.013454134396789596  -- Acc tensor(99.5300, device='cuda:0')
Epoch: 38 -- Phase:val -- Loss 1.0141414293289184  -- Acc tensor(83.5100, device='cuda:0')
Epoch: 39 -- Phase:train -- Loss 0.01615760260089766  -- Acc tensor(99.4460, device='cuda:0')
Epoch: 39 -- Phase:val -- Loss 1.0005965675354005  -- Acc tensor(83.5800, device='cuda:0')
"""