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
plt.savefig('/scratch/ph2200/pytorch-example/test_accuracy_model12')

plt.cla()
#loss
plt.plot(torch.tensor(save_loss['train']))
plt.plot(torch.tensor(save_loss['val']))
plt.legend(["train", "val"])
plt.title("Loss")    
plt.savefig('/scratch/ph2200/pytorch-example/test_loss_model12')
   
  

model_path = '/scratch/ph2200/pytorch-example/project1_model_12.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4,368,690
Epoch: 0 -- Phase:train -- Loss 1.4080904277801514  -- Acc tensor(48.5320, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.1404353511810303  -- Acc tensor(59.8900, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.936178761177063  -- Acc tensor(66.6980, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.9016648845672608  -- Acc tensor(69.2800, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.7381783867645264  -- Acc tensor(74.0720, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.7915279036521912  -- Acc tensor(73.3300, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.6256288668441773  -- Acc tensor(78.2780, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.6194202291488647  -- Acc tensor(79.5500, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.5494078997612  -- Acc tensor(81.0580, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.5623464527606964  -- Acc tensor(80.9600, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.4960776133441925  -- Acc tensor(82.7360, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.578285924577713  -- Acc tensor(80.1800, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.45019931480407716  -- Acc tensor(84.4560, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.492315114569664  -- Acc tensor(83.3900, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.41291827083587646  -- Acc tensor(85.7660, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.5588268530845643  -- Acc tensor(81.3400, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.3804073469543457  -- Acc tensor(86.7800, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.45365556092262266  -- Acc tensor(85.6300, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.3527368616962433  -- Acc tensor(87.7560, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.4164010542750359  -- Acc tensor(86.1500, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.32790276045799255  -- Acc tensor(88.6260, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.3858706425189972  -- Acc tensor(87.2000, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.305556270904541  -- Acc tensor(89.3920, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.3942662017047405  -- Acc tensor(87.5800, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.28604797881126404  -- Acc tensor(89.9220, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.3780182632446289  -- Acc tensor(87.8700, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.270449396944046  -- Acc tensor(90.6540, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.36755469324588774  -- Acc tensor(87.9700, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.2511360074424744  -- Acc tensor(91.2260, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.36231654210090636  -- Acc tensor(88.1300, device='cuda:0')"""