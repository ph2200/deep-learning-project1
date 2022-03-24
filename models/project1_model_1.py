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

#accuracy
plt.plot(torch.tensor(save_acc['train']))
plt.plot(torch.tensor(save_acc['val']))
plt.legend(["train", "val"])
plt.title("Accuracy")    
plt.savefig('/scratch/ph2200/pytorch-example/test_accuracy_model1')

#loss
plt.cla()
plt.plot(torch.tensor(save_loss['train']))
plt.plot(torch.tensor(save_loss['val']))
plt.legend(["train", "val"])
plt.title("Loss")    
plt.savefig('/scratch/ph2200/pytorch-example/test_loss_model1')
  
   

model_path = '/scratch/ph2200/pytorch-example/project1_model_1.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4,368,690
Epoch: 0 -- Phase:train -- Loss 1.269960568370819  -- Acc tensor(53.7260, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 0.9239732669830323  -- Acc tensor(67.6700, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.7649368343734742  -- Acc tensor(73.1380, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.7072596769332886  -- Acc tensor(75.2800, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.5774337714576722  -- Acc tensor(79.7400, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.6381012349128723  -- Acc tensor(77.9700, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.4546149560832977  -- Acc tensor(84.2360, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.5963286650657654  -- Acc tensor(80.1600, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.3597163941955566  -- Acc tensor(87.4620, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.5780942361354828  -- Acc tensor(80.8500, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.2670054637527466  -- Acc tensor(90.6720, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.6904366539001465  -- Acc tensor(79.1600, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.20637776871204377  -- Acc tensor(92.6280, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.589695294380188  -- Acc tensor(82.2500, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.14789519271612167  -- Acc tensor(94.8000, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.6268092594146728  -- Acc tensor(82.7100, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.11122393736958504  -- Acc tensor(96.0100, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.6665263912200928  -- Acc tensor(83.0900, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.09382451635420322  -- Acc tensor(96.6540, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.703583876991272  -- Acc tensor(82.6100, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.0791791023671627  -- Acc tensor(97.2440, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.7793706245422364  -- Acc tensor(81.6900, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.06666372008681297  -- Acc tensor(97.7300, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.7865608724594116  -- Acc tensor(82.4500, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.06345101320892572  -- Acc tensor(97.7460, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.8518337536811829  -- Acc tensor(81.3900, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.05310929205328226  -- Acc tensor(98.0900, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.809800648021698  -- Acc tensor(82.5100, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.04874854773826897  -- Acc tensor(98.3280, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.836566794013977  -- Acc tensor(82.6500, device='cuda:0')"""
