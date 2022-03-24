import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
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
plt.savefig('/scratch/ph2200/pytorch-example/test_accuracy_model2')

plt.cla()
#loss
plt.plot(torch.tensor(save_loss['train']))
plt.plot(torch.tensor(save_loss['val']))
plt.legend(["train", "val"])
plt.title("Loss")    
plt.savefig('/scratch/ph2200/pytorch-example/test_loss_model2')
     
  

model_path = '/scratch/ph2200/pytorch-example/project1_model_2.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4,368,690
Epoch: 0 -- Phase:train -- Loss 1.2435481937026978  -- Acc tensor(54.8360, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.0050954168319701  -- Acc tensor(64.6200, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.7865202885437012  -- Acc tensor(72.2680, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.7135588275909424  -- Acc tensor(75.3500, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.611340163230896  -- Acc tensor(78.6720, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.619851310825348  -- Acc tensor(78.7500, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.5102094574928284  -- Acc tensor(82.2700, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.5745145516395569  -- Acc tensor(80.3800, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.43063008281707765  -- Acc tensor(85.0780, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.6741429886817932  -- Acc tensor(78.4200, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.37595797327041625  -- Acc tensor(86.8920, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.4996224795341492  -- Acc tensor(83.3100, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.3239372623348236  -- Acc tensor(88.9520, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.49220797767639163  -- Acc tensor(83.1900, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.27725068027496336  -- Acc tensor(90.3960, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.482142227268219  -- Acc tensor(84.2000, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.2416601293706894  -- Acc tensor(91.6000, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.457791211271286  -- Acc tensor(85.6200, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.20659034118652345  -- Acc tensor(92.6960, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.4870304395675659  -- Acc tensor(84.8900, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.1789704630470276  -- Acc tensor(93.7400, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.47099853906631467  -- Acc tensor(86.1800, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.15453502567231656  -- Acc tensor(94.6560, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.4981939434528351  -- Acc tensor(85.3800, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.1311113770365715  -- Acc tensor(95.4620, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.5264801825046539  -- Acc tensor(85.4300, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.1190461345744133  -- Acc tensor(95.8200, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.5515740124702454  -- Acc tensor(85.2700, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.09859960943102837  -- Acc tensor(96.5380, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.507154132938385  -- Acc tensor(86.8200, device='cuda:0')"""