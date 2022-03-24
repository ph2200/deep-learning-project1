import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
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
plt.savefig('/scratch/ph2200/pytorch-example/test_accuracy_model3')

plt.cla()
#loss
plt.plot(torch.tensor(save_loss['train']))
plt.plot(torch.tensor(save_loss['val']))
plt.legend(["train", "val"])
plt.title("Loss")    
plt.savefig('/scratch/ph2200/pytorch-example/test_loss_model3')
     

model_path = '/scratch/ph2200/pytorch-example/project1_model_3.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4816890

Epoch: 0 -- Phase:train -- Loss 1.3782540478515626  -- Acc tensor(49.4640, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.0363232128143312  -- Acc tensor(63.3800, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.937135385761261  -- Acc tensor(66.7960, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.9768722501754761  -- Acc tensor(66.5500, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.7447560272216797  -- Acc tensor(73.9220, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.7733902467727661  -- Acc tensor(73.8500, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.6239841190719605  -- Acc tensor(78.4220, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.6628333249092102  -- Acc tensor(77.5800, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.5470730355834961  -- Acc tensor(80.9740, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.5960448196411133  -- Acc tensor(80.0700, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.4905170177268982  -- Acc tensor(83.1620, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.5754842589378357  -- Acc tensor(81.1400, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.4447966207695007  -- Acc tensor(84.6520, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.4869223834037781  -- Acc tensor(83.0300, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.406458376455307  -- Acc tensor(86.1460, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.4393780282020569  -- Acc tensor(85.3700, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.3679723182010651  -- Acc tensor(87.2940, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.4616286997795105  -- Acc tensor(84.9100, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.3442023352622986  -- Acc tensor(88.0420, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.40819372148513794  -- Acc tensor(86.6800, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.31904594223976135  -- Acc tensor(89.0820, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.3942509711265564  -- Acc tensor(87.2100, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.29630779482364655  -- Acc tensor(89.7120, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.42623146729469297  -- Acc tensor(86.2400, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.27566616302013397  -- Acc tensor(90.4380, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.37686064682006837  -- Acc tensor(87.3700, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.2595114413070679  -- Acc tensor(91.0800, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.36239604041576384  -- Acc tensor(88.1200, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.24252362566947938  -- Acc tensor(91.5080, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.4067880262374878  -- Acc tensor(87.2700, device='cuda:0')"""