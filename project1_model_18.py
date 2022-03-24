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
plt.savefig('/scratch/ph2200/pytorch-example/test_accuracy_model18')

plt.cla()
#loss
plt.plot(torch.tensor(save_loss['train']))
plt.plot(torch.tensor(save_loss['val']))
plt.legend(["train", "val"])
plt.title("Loss")    
plt.savefig('/scratch/ph2200/pytorch-example/test_loss_model18')
   
  

model_path = '/scratch/ph2200/pytorch-example/project1_model_18.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4,368,690

Epoch: 0 -- Phase:train -- Loss 1.4361145058059692  -- Acc tensor(47.1380, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.0996609125137329  -- Acc tensor(60.7700, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.9590915272521973  -- Acc tensor(65.7000, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 1.0424015909194946  -- Acc tensor(64.2300, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.7516105784225464  -- Acc tensor(73.4080, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.6799461748123169  -- Acc tensor(76.5800, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.6316745750808715  -- Acc tensor(77.8000, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.6605410304069519  -- Acc tensor(77.7200, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.5503295765304566  -- Acc tensor(80.9220, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.5817911571979523  -- Acc tensor(80.4600, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.49530349605560303  -- Acc tensor(82.7700, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.5220266863822937  -- Acc tensor(82.3200, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.4436040764808655  -- Acc tensor(84.7380, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.4874402479171753  -- Acc tensor(84.0700, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.4048363761806488  -- Acc tensor(86.2040, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.4437787397384644  -- Acc tensor(85.3600, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.3794897126865387  -- Acc tensor(86.9800, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.4502904028892517  -- Acc tensor(85.4100, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.35076216099739077  -- Acc tensor(87.8980, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.43907663860321045  -- Acc tensor(85.4100, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.32139771689414975  -- Acc tensor(88.7960, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.3919919970035553  -- Acc tensor(87.1100, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.29598189261436464  -- Acc tensor(89.6780, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.3999577254056931  -- Acc tensor(87.3600, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.278578853931427  -- Acc tensor(90.3200, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.3697372269153595  -- Acc tensor(87.8500, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.26512180963516235  -- Acc tensor(90.8840, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.3504734768867493  -- Acc tensor(88.8100, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.2437129024887085  -- Acc tensor(91.4640, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.37246369812488556  -- Acc tensor(88.4100, device='cuda:0')
Epoch: 15 -- Phase:train -- Loss 0.2298959874200821  -- Acc tensor(92.0020, device='cuda:0')
Epoch: 15 -- Phase:val -- Loss 0.33494815454483035  -- Acc tensor(89.2100, device='cuda:0')
Epoch: 16 -- Phase:train -- Loss 0.2190579230928421  -- Acc tensor(92.2420, device='cuda:0')
Epoch: 16 -- Phase:val -- Loss 0.3587260237693787  -- Acc tensor(88.7300, device='cuda:0')
Epoch: 17 -- Phase:train -- Loss 0.2035818126153946  -- Acc tensor(92.8440, device='cuda:0')
Epoch: 17 -- Phase:val -- Loss 0.35743173174858095  -- Acc tensor(88.8700, device='cuda:0')
Epoch: 18 -- Phase:train -- Loss 0.1915099640750885  -- Acc tensor(93.2420, device='cuda:0')
Epoch: 18 -- Phase:val -- Loss 0.3671419244766235  -- Acc tensor(88.7900, device='cuda:0')
Epoch: 19 -- Phase:train -- Loss 0.18172011703491212  -- Acc tensor(93.5780, device='cuda:0')
Epoch: 19 -- Phase:val -- Loss 0.3634690739631653  -- Acc tensor(89.3500, device='cuda:0')
Epoch: 20 -- Phase:train -- Loss 0.17355292655467988  -- Acc tensor(93.9180, device='cuda:0')
Epoch: 20 -- Phase:val -- Loss 0.3474315106630325  -- Acc tensor(89.8500, device='cuda:0')
Epoch: 21 -- Phase:train -- Loss 0.1618373535346985  -- Acc tensor(94.2840, device='cuda:0')
Epoch: 21 -- Phase:val -- Loss 0.3558209129810333  -- Acc tensor(89.5700, device='cuda:0')
Epoch: 22 -- Phase:train -- Loss 0.15204058277130128  -- Acc tensor(94.6260, device='cuda:0')
Epoch: 22 -- Phase:val -- Loss 0.38271624417304995  -- Acc tensor(88.7300, device='cuda:0')
Epoch: 23 -- Phase:train -- Loss 0.14736289419174195  -- Acc tensor(94.7400, device='cuda:0')
Epoch: 23 -- Phase:val -- Loss 0.32290320953130724  -- Acc tensor(90.5200, device='cuda:0')
Epoch: 24 -- Phase:train -- Loss 0.1396059018945694  -- Acc tensor(95.0480, device='cuda:0')
Epoch: 24 -- Phase:val -- Loss 0.367098015832901  -- Acc tensor(89.4200, device='cuda:0')
Epoch: 25 -- Phase:train -- Loss 0.13200258713483812  -- Acc tensor(95.3020, device='cuda:0')
Epoch: 25 -- Phase:val -- Loss 0.37955234515666963  -- Acc tensor(89.4700, device='cuda:0')
Epoch: 26 -- Phase:train -- Loss 0.12470325488805771  -- Acc tensor(95.6400, device='cuda:0')
Epoch: 26 -- Phase:val -- Loss 0.3676057103157043  -- Acc tensor(89.5900, device='cuda:0')
Epoch: 27 -- Phase:train -- Loss 0.11756934192895889  -- Acc tensor(95.8520, device='cuda:0')
Epoch: 27 -- Phase:val -- Loss 0.39365006692409515  -- Acc tensor(89.8400, device='cuda:0')
Epoch: 28 -- Phase:train -- Loss 0.1163724053800106  -- Acc tensor(95.8700, device='cuda:0')
Epoch: 28 -- Phase:val -- Loss 0.3540801803588867  -- Acc tensor(90.3200, device='cuda:0')
Epoch: 29 -- Phase:train -- Loss 0.10726160726308823  -- Acc tensor(96.2400, device='cuda:0')
Epoch: 29 -- Phase:val -- Loss 0.35516410088539124  -- Acc tensor(90.2200, device='cuda:0')
Epoch: 30 -- Phase:train -- Loss 0.10405624123930932  -- Acc tensor(96.2540, device='cuda:0')
Epoch: 30 -- Phase:val -- Loss 0.3300021170139313  -- Acc tensor(90.8900, device='cuda:0')
Epoch: 31 -- Phase:train -- Loss 0.10040903212428093  -- Acc tensor(96.4680, device='cuda:0')
Epoch: 31 -- Phase:val -- Loss 0.39177007665634156  -- Acc tensor(89.6700, device='cuda:0')
Epoch: 32 -- Phase:train -- Loss 0.09328664367616177  -- Acc tensor(96.7540, device='cuda:0')
Epoch: 32 -- Phase:val -- Loss 0.36463843841552734  -- Acc tensor(90.6200, device='cuda:0')
Epoch: 33 -- Phase:train -- Loss 0.08682056926578283  -- Acc tensor(96.9440, device='cuda:0')
Epoch: 33 -- Phase:val -- Loss 0.3572069440484047  -- Acc tensor(90.8200, device='cuda:0')
Epoch: 34 -- Phase:train -- Loss 0.08802876738518477  -- Acc tensor(96.9540, device='cuda:0')
Epoch: 34 -- Phase:val -- Loss 0.38394844553470614  -- Acc tensor(90.4400, device='cuda:0')
Epoch: 35 -- Phase:train -- Loss 0.08278855729550123  -- Acc tensor(97.0480, device='cuda:0')
Epoch: 35 -- Phase:val -- Loss 0.37566942114830015  -- Acc tensor(90.4700, device='cuda:0')
Epoch: 36 -- Phase:train -- Loss 0.08064186692297459  -- Acc tensor(97.1840, device='cuda:0')
Epoch: 36 -- Phase:val -- Loss 0.35689975382089617  -- Acc tensor(91.1100, device='cuda:0')
Epoch: 37 -- Phase:train -- Loss 0.07852494919121265  -- Acc tensor(97.1580, device='cuda:0')
Epoch: 37 -- Phase:val -- Loss 0.39985058728456496  -- Acc tensor(90.5400, device='cuda:0')
Epoch: 38 -- Phase:train -- Loss 0.0711021842521429  -- Acc tensor(97.5200, device='cuda:0')
Epoch: 38 -- Phase:val -- Loss 0.41664471747875215  -- Acc tensor(90., device='cuda:0')
Epoch: 39 -- Phase:train -- Loss 0.07452920032203197  -- Acc tensor(97.3180, device='cuda:0')
Epoch: 39 -- Phase:val -- Loss 0.39737276487350465  -- Acc tensor(90.7400, device='cuda:0')
"""