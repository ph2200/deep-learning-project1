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


for epoch in range(70):

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

    
  

model_path = '/scratch/ph2200/pytorch-example/project1_model_8.pt'
torch.save(model.state_dict(), model_path)

"""
# of parameters=4816890

Epoch: 0 -- Phase:train -- Loss 1.40615258808136  -- Acc tensor(48.6060, device='cuda:0')
Epoch: 0 -- Phase:val -- Loss 1.0907119520187378  -- Acc tensor(61.1800, device='cuda:0')
Epoch: 1 -- Phase:train -- Loss 0.9400783325195312  -- Acc tensor(66.6380, device='cuda:0')
Epoch: 1 -- Phase:val -- Loss 0.9313074407577515  -- Acc tensor(68.6700, device='cuda:0')
Epoch: 2 -- Phase:train -- Loss 0.7406692712211609  -- Acc tensor(74.1680, device='cuda:0')
Epoch: 2 -- Phase:val -- Loss 0.6553989298820495  -- Acc tensor(77.7500, device='cuda:0')
Epoch: 3 -- Phase:train -- Loss 0.6229636482048034  -- Acc tensor(78.5540, device='cuda:0')
Epoch: 3 -- Phase:val -- Loss 0.6323085362434387  -- Acc tensor(78.8000, device='cuda:0')
Epoch: 4 -- Phase:train -- Loss 0.538415969657898  -- Acc tensor(81.3040, device='cuda:0')
Epoch: 4 -- Phase:val -- Loss 0.6474783228874207  -- Acc tensor(78.3400, device='cuda:0')
Epoch: 5 -- Phase:train -- Loss 0.48771780588150027  -- Acc tensor(83.2560, device='cuda:0')
Epoch: 5 -- Phase:val -- Loss 0.4896586763858795  -- Acc tensor(83.6800, device='cuda:0')
Epoch: 6 -- Phase:train -- Loss 0.4383108423233032  -- Acc tensor(84.9200, device='cuda:0')
Epoch: 6 -- Phase:val -- Loss 0.4927889995574951  -- Acc tensor(83.7500, device='cuda:0')
Epoch: 7 -- Phase:train -- Loss 0.39911943850517273  -- Acc tensor(86.3140, device='cuda:0')
Epoch: 7 -- Phase:val -- Loss 0.4316799365520477  -- Acc tensor(85.6500, device='cuda:0')
Epoch: 8 -- Phase:train -- Loss 0.3684249479579926  -- Acc tensor(87.3780, device='cuda:0')
Epoch: 8 -- Phase:val -- Loss 0.44339821164608  -- Acc tensor(85.4000, device='cuda:0')
Epoch: 9 -- Phase:train -- Loss 0.3406746985292435  -- Acc tensor(88.1120, device='cuda:0')
Epoch: 9 -- Phase:val -- Loss 0.38301026754379275  -- Acc tensor(87.0600, device='cuda:0')
Epoch: 10 -- Phase:train -- Loss 0.3161102367687225  -- Acc tensor(88.9760, device='cuda:0')
Epoch: 10 -- Phase:val -- Loss 0.3870564861655235  -- Acc tensor(87.2400, device='cuda:0')
Epoch: 11 -- Phase:train -- Loss 0.29339262558698653  -- Acc tensor(89.7880, device='cuda:0')
Epoch: 11 -- Phase:val -- Loss 0.38626606121063234  -- Acc tensor(87.7800, device='cuda:0')
Epoch: 12 -- Phase:train -- Loss 0.27046191162347794  -- Acc tensor(90.6280, device='cuda:0')
Epoch: 12 -- Phase:val -- Loss 0.3580584617137909  -- Acc tensor(88.7700, device='cuda:0')
Epoch: 13 -- Phase:train -- Loss 0.25381885236263274  -- Acc tensor(91.1340, device='cuda:0')
Epoch: 13 -- Phase:val -- Loss 0.3672306258201599  -- Acc tensor(88.0400, device='cuda:0')
Epoch: 14 -- Phase:train -- Loss 0.23902626391887666  -- Acc tensor(91.6460, device='cuda:0')
Epoch: 14 -- Phase:val -- Loss 0.3403001991152763  -- Acc tensor(88.9700, device='cuda:0')
Epoch: 15 -- Phase:train -- Loss 0.22090267043590545  -- Acc tensor(92.3240, device='cuda:0')
Epoch: 15 -- Phase:val -- Loss 0.36267991502285  -- Acc tensor(88.8900, device='cuda:0')
Epoch: 16 -- Phase:train -- Loss 0.20958506713867187  -- Acc tensor(92.5980, device='cuda:0')
Epoch: 16 -- Phase:val -- Loss 0.353846772813797  -- Acc tensor(89.0700, device='cuda:0')
Epoch: 17 -- Phase:train -- Loss 0.19604457556724547  -- Acc tensor(93.2320, device='cuda:0')
Epoch: 17 -- Phase:val -- Loss 0.356180832862854  -- Acc tensor(89.1200, device='cuda:0')
Epoch: 18 -- Phase:train -- Loss 0.18478455101013183  -- Acc tensor(93.5340, device='cuda:0')
Epoch: 18 -- Phase:val -- Loss 0.35489446535110475  -- Acc tensor(89.6600, device='cuda:0')
Epoch: 19 -- Phase:train -- Loss 0.1743889341068268  -- Acc tensor(93.8440, device='cuda:0')
Epoch: 19 -- Phase:val -- Loss 0.3427207732200623  -- Acc tensor(89.9500, device='cuda:0')
Epoch: 20 -- Phase:train -- Loss 0.16605457870006562  -- Acc tensor(94.1900, device='cuda:0')
Epoch: 20 -- Phase:val -- Loss 0.33424788110256193  -- Acc tensor(89.7200, device='cuda:0')
Epoch: 21 -- Phase:train -- Loss 0.15507451698064803  -- Acc tensor(94.5100, device='cuda:0')
Epoch: 21 -- Phase:val -- Loss 0.31463311855792997  -- Acc tensor(90.9000, device='cuda:0')
Epoch: 22 -- Phase:train -- Loss 0.14619454286813735  -- Acc tensor(94.7840, device='cuda:0')
Epoch: 22 -- Phase:val -- Loss 0.3396801801919937  -- Acc tensor(90.1900, device='cuda:0')
Epoch: 23 -- Phase:train -- Loss 0.1369201234149933  -- Acc tensor(95.2060, device='cuda:0')
Epoch: 23 -- Phase:val -- Loss 0.34108558027744296  -- Acc tensor(90.2000, device='cuda:0')
Epoch: 24 -- Phase:train -- Loss 0.12938465828180312  -- Acc tensor(95.4760, device='cuda:0')
Epoch: 24 -- Phase:val -- Loss 0.3413938055992127  -- Acc tensor(90.3600, device='cuda:0')
Epoch: 25 -- Phase:train -- Loss 0.12574545578360558  -- Acc tensor(95.5820, device='cuda:0')
Epoch: 25 -- Phase:val -- Loss 0.3470576856017113  -- Acc tensor(90.6000, device='cuda:0')
Epoch: 26 -- Phase:train -- Loss 0.1174333317232132  -- Acc tensor(95.8960, device='cuda:0')
Epoch: 26 -- Phase:val -- Loss 0.3594170015573502  -- Acc tensor(89.9200, device='cuda:0')
Epoch: 27 -- Phase:train -- Loss 0.11291757276535035  -- Acc tensor(96.0180, device='cuda:0')
Epoch: 27 -- Phase:val -- Loss 0.33279995790719985  -- Acc tensor(90.8100, device='cuda:0')
Epoch: 28 -- Phase:train -- Loss 0.10351653135299682  -- Acc tensor(96.2880, device='cuda:0')
Epoch: 28 -- Phase:val -- Loss 0.35961408904790876  -- Acc tensor(90.5200, device='cuda:0')
Epoch: 29 -- Phase:train -- Loss 0.10139208423256874  -- Acc tensor(96.3820, device='cuda:0')
Epoch: 29 -- Phase:val -- Loss 0.3616115265369415  -- Acc tensor(90.7600, device='cuda:0')
Epoch: 30 -- Phase:train -- Loss 0.0968894277408719  -- Acc tensor(96.5460, device='cuda:0')
Epoch: 30 -- Phase:val -- Loss 0.3518239937067032  -- Acc tensor(90.6100, device='cuda:0')
Epoch: 31 -- Phase:train -- Loss 0.09271535408556461  -- Acc tensor(96.6200, device='cuda:0')
Epoch: 31 -- Phase:val -- Loss 0.3734832114458084  -- Acc tensor(90.5200, device='cuda:0')
Epoch: 32 -- Phase:train -- Loss 0.09114489440977573  -- Acc tensor(96.8320, device='cuda:0')
Epoch: 32 -- Phase:val -- Loss 0.36863286757469177  -- Acc tensor(90.7800, device='cuda:0')
Epoch: 33 -- Phase:train -- Loss 0.0882629350489378  -- Acc tensor(96.8360, device='cuda:0')
Epoch: 33 -- Phase:val -- Loss 0.35864472143650056  -- Acc tensor(90.8100, device='cuda:0')
Epoch: 34 -- Phase:train -- Loss 0.0766091932553053  -- Acc tensor(97.2480, device='cuda:0')
Epoch: 34 -- Phase:val -- Loss 0.373944872713089  -- Acc tensor(91.1800, device='cuda:0')
Epoch: 35 -- Phase:train -- Loss 0.08251538862586022  -- Acc tensor(97.0320, device='cuda:0')
Epoch: 35 -- Phase:val -- Loss 0.39577799863815305  -- Acc tensor(90.6400, device='cuda:0')
Epoch: 36 -- Phase:train -- Loss 0.07524376070052385  -- Acc tensor(97.3620, device='cuda:0')
Epoch: 36 -- Phase:val -- Loss 0.3819869595527649  -- Acc tensor(90.7200, device='cuda:0')
Epoch: 37 -- Phase:train -- Loss 0.07241799633920193  -- Acc tensor(97.4680, device='cuda:0')
Epoch: 37 -- Phase:val -- Loss 0.3861335210800171  -- Acc tensor(91.1500, device='cuda:0')
Epoch: 38 -- Phase:train -- Loss 0.07036341000288725  -- Acc tensor(97.4960, device='cuda:0')
Epoch: 38 -- Phase:val -- Loss 0.3694026053667068  -- Acc tensor(91.2900, device='cuda:0')
Epoch: 39 -- Phase:train -- Loss 0.06851501524567605  -- Acc tensor(97.5160, device='cuda:0')
Epoch: 39 -- Phase:val -- Loss 0.36690260075330733  -- Acc tensor(91.5000, device='cuda:0')
Epoch: 40 -- Phase:train -- Loss 0.06605149395108223  -- Acc tensor(97.6620, device='cuda:0')
Epoch: 40 -- Phase:val -- Loss 0.39439926015138627  -- Acc tensor(91.2400, device='cuda:0')
Epoch: 41 -- Phase:train -- Loss 0.0629676178663969  -- Acc tensor(97.7920, device='cuda:0')
Epoch: 41 -- Phase:val -- Loss 0.3644364299535751  -- Acc tensor(91.4600, device='cuda:0')
Epoch: 42 -- Phase:train -- Loss 0.06232475938111544  -- Acc tensor(97.7580, device='cuda:0')
Epoch: 42 -- Phase:val -- Loss 0.41433048739433287  -- Acc tensor(90.7000, device='cuda:0')
Epoch: 43 -- Phase:train -- Loss 0.06160214664846659  -- Acc tensor(97.7960, device='cuda:0')
Epoch: 43 -- Phase:val -- Loss 0.39082110891342164  -- Acc tensor(91.0100, device='cuda:0')
Epoch: 44 -- Phase:train -- Loss 0.05953099761992693  -- Acc tensor(97.8740, device='cuda:0')
Epoch: 44 -- Phase:val -- Loss 0.37609251841306685  -- Acc tensor(91.4900, device='cuda:0')
Epoch: 45 -- Phase:train -- Loss 0.05455742133788764  -- Acc tensor(98.0900, device='cuda:0')
Epoch: 45 -- Phase:val -- Loss 0.406198407125473  -- Acc tensor(91.2600, device='cuda:0')
Epoch: 46 -- Phase:train -- Loss 0.05351454767659306  -- Acc tensor(98.1680, device='cuda:0')
Epoch: 46 -- Phase:val -- Loss 0.39903890486955645  -- Acc tensor(91.3000, device='cuda:0')
Epoch: 47 -- Phase:train -- Loss 0.05284694533675909  -- Acc tensor(98.1600, device='cuda:0')
Epoch: 47 -- Phase:val -- Loss 0.398162274479866  -- Acc tensor(91.2800, device='cuda:0')
Epoch: 48 -- Phase:train -- Loss 0.05093890965789556  -- Acc tensor(98.2700, device='cuda:0')
Epoch: 48 -- Phase:val -- Loss 0.38763328235149386  -- Acc tensor(91.3300, device='cuda:0')
Epoch: 49 -- Phase:train -- Loss 0.05174833685168997  -- Acc tensor(98.2200, device='cuda:0')
Epoch: 49 -- Phase:val -- Loss 0.36688484697341917  -- Acc tensor(91.7000, device='cuda:0')
Epoch: 50 -- Phase:train -- Loss 0.04645657214850187  -- Acc tensor(98.3680, device='cuda:0')
Epoch: 50 -- Phase:val -- Loss 0.3927413124322891  -- Acc tensor(91.4700, device='cuda:0')
Epoch: 51 -- Phase:train -- Loss 0.05132400716871023  -- Acc tensor(98.1880, device='cuda:0')
Epoch: 51 -- Phase:val -- Loss 0.4178120623826981  -- Acc tensor(91.2100, device='cuda:0')
Epoch: 52 -- Phase:train -- Loss 0.04903225131228566  -- Acc tensor(98.2880, device='cuda:0')
Epoch: 52 -- Phase:val -- Loss 0.4139233713746071  -- Acc tensor(91.2300, device='cuda:0')
Epoch: 53 -- Phase:train -- Loss 0.04812358511596918  -- Acc tensor(98.3300, device='cuda:0')
Epoch: 53 -- Phase:val -- Loss 0.40316140706539155  -- Acc tensor(91.2000, device='cuda:0')
Epoch: 54 -- Phase:train -- Loss 0.042217146367281674  -- Acc tensor(98.5000, device='cuda:0')
Epoch: 54 -- Phase:val -- Loss 0.44259876246452334  -- Acc tensor(90.9200, device='cuda:0')
Epoch: 55 -- Phase:train -- Loss 0.0452833835542202  -- Acc tensor(98.4200, device='cuda:0')
Epoch: 55 -- Phase:val -- Loss 0.38194397240877154  -- Acc tensor(91.8000, device='cuda:0')
Epoch: 56 -- Phase:train -- Loss 0.04415247487977147  -- Acc tensor(98.4340, device='cuda:0')
Epoch: 56 -- Phase:val -- Loss 0.4533691423341632  -- Acc tensor(91.0100, device='cuda:0')
Epoch: 57 -- Phase:train -- Loss 0.04389648808598518  -- Acc tensor(98.4660, device='cuda:0')
Epoch: 57 -- Phase:val -- Loss 0.3824557313680649  -- Acc tensor(91.8500, device='cuda:0')
Epoch: 58 -- Phase:train -- Loss 0.040076279279589656  -- Acc tensor(98.6140, device='cuda:0')
Epoch: 58 -- Phase:val -- Loss 0.4340125789105892  -- Acc tensor(91.1100, device='cuda:0')
Epoch: 59 -- Phase:train -- Loss 0.04443184090415016  -- Acc tensor(98.4760, device='cuda:0')
Epoch: 59 -- Phase:val -- Loss 0.4271043751001358  -- Acc tensor(91.1000, device='cuda:0')
Epoch: 60 -- Phase:train -- Loss 0.035809647380411626  -- Acc tensor(98.7340, device='cuda:0')
Epoch: 60 -- Phase:val -- Loss 0.41579028171002863  -- Acc tensor(91.5500, device='cuda:0')
Epoch: 61 -- Phase:train -- Loss 0.03942770209483802  -- Acc tensor(98.6660, device='cuda:0')
Epoch: 61 -- Phase:val -- Loss 0.41505678679645064  -- Acc tensor(91.5100, device='cuda:0')
Epoch: 62 -- Phase:train -- Loss 0.038244257752522826  -- Acc tensor(98.6460, device='cuda:0')
Epoch: 62 -- Phase:val -- Loss 0.42836645004749296  -- Acc tensor(91.2700, device='cuda:0')
Epoch: 63 -- Phase:train -- Loss 0.03790760221995413  -- Acc tensor(98.6840, device='cuda:0')
Epoch: 63 -- Phase:val -- Loss 0.39756943482160567  -- Acc tensor(91.7300, device='cuda:0')
Epoch: 64 -- Phase:train -- Loss 0.036736127742156385  -- Acc tensor(98.6580, device='cuda:0')
Epoch: 64 -- Phase:val -- Loss 0.42353091652393343  -- Acc tensor(91.5800, device='cuda:0')
Epoch: 65 -- Phase:train -- Loss 0.034266700149923564  -- Acc tensor(98.8760, device='cuda:0')
Epoch: 65 -- Phase:val -- Loss 0.44786819755733015  -- Acc tensor(90.9900, device='cuda:0')
Epoch: 66 -- Phase:train -- Loss 0.03674940999567509  -- Acc tensor(98.7640, device='cuda:0')
Epoch: 66 -- Phase:val -- Loss 0.43284023718833925  -- Acc tensor(91.4700, device='cuda:0')
Epoch: 67 -- Phase:train -- Loss 0.037061975490562615  -- Acc tensor(98.6920, device='cuda:0')
Epoch: 67 -- Phase:val -- Loss 0.4214497758567333  -- Acc tensor(91.7900, device='cuda:0')
Epoch: 68 -- Phase:train -- Loss 0.03262581769650802  -- Acc tensor(98.9020, device='cuda:0')
Epoch: 68 -- Phase:val -- Loss 0.4518493003964424  -- Acc tensor(91.7100, device='cuda:0')
Epoch: 69 -- Phase:train -- Loss 0.03334218436470255  -- Acc tensor(98.8100, device='cuda:0')
Epoch: 69 -- Phase:val -- Loss 0.4881160573244095  -- Acc tensor(90.8800, device='cuda:0')
"""