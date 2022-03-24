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
plt.savefig('/scratch/ph2200/pytorch-example/test_accuracy_model4')

#loss
plt.cla()
plt.plot(torch.tensor(save_loss['train']))
plt.plot(torch.tensor(save_loss['val']))
plt.legend(["train", "val"])
plt.title("Loss")    
plt.savefig('/scratch/ph2200/pytorch-example/test_loss_model4')
  
model_path = '/scratch/ph2200/pytorch-example/project1_model_4.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4816890

Epoch: 0 -- Phase:train -- Loss 1.3866398723983764  -- Acc tensor(49.0000, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.260016937828064  -- Acc tensor(57.3900, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.93584089427948  -- Acc tensor(66.8040, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.8673712924003601  -- Acc tensor(70.0800, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.7463640096473694  -- Acc tensor(73.8680, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.7043638670921326  -- Acc tensor(76.2400, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.6346752286624908  -- Acc tensor(77.9120, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.6419163453102111  -- Acc tensor(78.1700, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.5550136860084534  -- Acc tensor(80.7080, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.6064427186012268  -- Acc tensor(79.7700, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.5012007006454468  -- Acc tensor(82.8260, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.5228874530792237  -- Acc tensor(82.5500, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.4523776331520081  -- Acc tensor(84.3900, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.4811565969944  -- Acc tensor(83.8400, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.4119721377182007  -- Acc tensor(85.7360, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.45858881311416627  -- Acc tensor(84.7500, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.3828030601501465  -- Acc tensor(86.8780, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.43695983176231384  -- Acc tensor(85.3300, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.35091938386917115  -- Acc tensor(87.8400, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.501951740026474  -- Acc tensor(83.9300, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.328880425567627  -- Acc tensor(88.5820, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.41139080295562747  -- Acc tensor(86.3400, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.3051785697174072  -- Acc tensor(89.4220, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.44168030090332033  -- Acc tensor(85.5400, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.27584832929968833  -- Acc tensor(90.3100, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.3647747157096863  -- Acc tensor(88.2200, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.26281633810043337  -- Acc tensor(90.9400, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.3677424560070038  -- Acc tensor(87.9500, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.24475756601810456  -- Acc tensor(91.5760, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.4097952298164368  -- Acc tensor(87.6300, device='cuda:0')
Epoch: 15 -- Phase:train -- Loss 0.23122344955444335  -- Acc tensor(91.8940, device='cuda:0')
Epoch: 15 -- Phase:val -- Loss 0.40270923104286194  -- Acc tensor(87.8200, device='cuda:0')
Epoch: 16 -- Phase:train -- Loss 0.21024734990596772  -- Acc tensor(92.6940, device='cuda:0')
Epoch: 16 -- Phase:val -- Loss 0.33477457752227785  -- Acc tensor(89.4000, device='cuda:0')
Epoch: 17 -- Phase:train -- Loss 0.20306058485984801  -- Acc tensor(92.7900, device='cuda:0')
Epoch: 17 -- Phase:val -- Loss 0.34699505529403685  -- Acc tensor(89.0800, device='cuda:0')
Epoch: 18 -- Phase:train -- Loss 0.19082638838052748  -- Acc tensor(93.3140, device='cuda:0')
Epoch: 18 -- Phase:val -- Loss 0.3857648005723953  -- Acc tensor(87.9600, device='cuda:0')
Epoch: 19 -- Phase:train -- Loss 0.17914496134757996  -- Acc tensor(93.6700, device='cuda:0')
Epoch: 19 -- Phase:val -- Loss 0.34985358328819277  -- Acc tensor(89.2900, device='cuda:0')
Epoch: 20 -- Phase:train -- Loss 0.17301482414484023  -- Acc tensor(93.8180, device='cuda:0')
Epoch: 20 -- Phase:val -- Loss 0.33769750604629517  -- Acc tensor(89.8100, device='cuda:0')
Epoch: 21 -- Phase:train -- Loss 0.15771793770313264  -- Acc tensor(94.2700, device='cuda:0')
Epoch: 21 -- Phase:val -- Loss 0.3770671018600464  -- Acc tensor(89.4400, device='cuda:0')
Epoch: 22 -- Phase:train -- Loss 0.14877470075368882  -- Acc tensor(94.7440, device='cuda:0')
Epoch: 22 -- Phase:val -- Loss 0.33094778814315795  -- Acc tensor(90.1600, device='cuda:0')
Epoch: 23 -- Phase:train -- Loss 0.14338043911337853  -- Acc tensor(94.9080, device='cuda:0')
Epoch: 23 -- Phase:val -- Loss 0.3424237509727478  -- Acc tensor(89.8900, device='cuda:0')
Epoch: 24 -- Phase:train -- Loss 0.13597906769990922  -- Acc tensor(95.2240, device='cuda:0')
Epoch: 24 -- Phase:val -- Loss 0.3459843488693237  -- Acc tensor(89.8500, device='cuda:0')
Epoch: 25 -- Phase:train -- Loss 0.12828176547050477  -- Acc tensor(95.4480, device='cuda:0')
Epoch: 25 -- Phase:val -- Loss 0.3470235367774963  -- Acc tensor(90.2300, device='cuda:0')
Epoch: 26 -- Phase:train -- Loss 0.12591821097135544  -- Acc tensor(95.5000, device='cuda:0')
Epoch: 26 -- Phase:val -- Loss 0.32481026971340177  -- Acc tensor(90.6400, device='cuda:0')
Epoch: 27 -- Phase:train -- Loss 0.11346317936301231  -- Acc tensor(95.9640, device='cuda:0')
Epoch: 27 -- Phase:val -- Loss 0.33606278283596036  -- Acc tensor(90.7500, device='cuda:0')
Epoch: 28 -- Phase:train -- Loss 0.11157312663793564  -- Acc tensor(96.0680, device='cuda:0')
Epoch: 28 -- Phase:val -- Loss 0.34015881280899046  -- Acc tensor(90.7700, device='cuda:0')
Epoch: 29 -- Phase:train -- Loss 0.10289714373707771  -- Acc tensor(96.2340, device='cuda:0')
Epoch: 29 -- Phase:val -- Loss 0.36317927939891814  -- Acc tensor(90.4700, device='cuda:0')
Epoch: 30 -- Phase:train -- Loss 0.09737859451532364  -- Acc tensor(96.6000, device='cuda:0')
Epoch: 30 -- Phase:val -- Loss 0.3746204959392548  -- Acc tensor(90.5800, device='cuda:0')
Epoch: 31 -- Phase:train -- Loss 0.10056666181385517  -- Acc tensor(96.4360, device='cuda:0')
Epoch: 31 -- Phase:val -- Loss 0.3397067405939102  -- Acc tensor(90.8600, device='cuda:0')
Epoch: 32 -- Phase:train -- Loss 0.09340886449038982  -- Acc tensor(96.7760, device='cuda:0')
Epoch: 32 -- Phase:val -- Loss 0.352722545671463  -- Acc tensor(91.1500, device='cuda:0')
Epoch: 33 -- Phase:train -- Loss 0.08751704188406467  -- Acc tensor(96.9180, device='cuda:0')
Epoch: 33 -- Phase:val -- Loss 0.33473415207862856  -- Acc tensor(91.4000, device='cuda:0')
Epoch: 34 -- Phase:train -- Loss 0.08406822420299054  -- Acc tensor(97.0520, device='cuda:0')
Epoch: 34 -- Phase:val -- Loss 0.35668053328990934  -- Acc tensor(90.9100, device='cuda:0')
Epoch: 35 -- Phase:train -- Loss 0.08374458520293236  -- Acc tensor(97.0440, device='cuda:0')
Epoch: 35 -- Phase:val -- Loss 0.3818198267459869  -- Acc tensor(90.4900, device='cuda:0')
Epoch: 36 -- Phase:train -- Loss 0.07621856984704733  -- Acc tensor(97.3100, device='cuda:0')
Epoch: 36 -- Phase:val -- Loss 0.3616324465751648  -- Acc tensor(91.0900, device='cuda:0')
Epoch: 37 -- Phase:train -- Loss 0.07627737384617328  -- Acc tensor(97.2840, device='cuda:0')
Epoch: 37 -- Phase:val -- Loss 0.3632126075088978  -- Acc tensor(91.1100, device='cuda:0')
Epoch: 38 -- Phase:train -- Loss 0.07444345195174218  -- Acc tensor(97.3900, device='cuda:0')
Epoch: 38 -- Phase:val -- Loss 0.4014493435740471  -- Acc tensor(90.5400, device='cuda:0')
Epoch: 39 -- Phase:train -- Loss 0.07171349070072174  -- Acc tensor(97.4900, device='cuda:0')
Epoch: 39 -- Phase:val -- Loss 0.3877016698598862  -- Acc tensor(90.8200, device='cuda:0')
"""
