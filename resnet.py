import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
from gumbelmodule import *
from layers import *

class Sequential_ext(nn.Module):
    def __init__(self, *args):
        super(Sequential_ext, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input, temperature=1, openings=None):
        gate_activations = []
        for i, module in enumerate(self._modules.values()):
            input, gate_activation = module(input, temperature)
            gate_activations.append(gate_activation)
        return input, gate_activations

def split(x,g):
    n=int(x.size()[1])
    n1 = round(n/g)
    xs=[]
    for i in range(g):
        xs.append(x[:,i*n1:(i+1)*n1,:,:].contiguous())
    # x1 = x[:, :n1, :, :].contiguous()
    # x2 = x[:, n1:, :, :].contiguous()
    return xs

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        out=x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)
        return out

def merge(xs):
    x0=xs[0]
    for x in xs[1:]:
        x0=torch.cat((x0,x),dim=1)
    return x0

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Weighted_Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1,expansion=1):
        super(Weighted_Bottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # Gate layers
        self.fc1 = nn.Conv2d(in_planes, 16, kernel_size=1)
        self.fc1bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2

        self.gs=GumbleSoftmax()
        self.gs.cuda()

    def forward(self, x, temperature=1):
        w = F.avg_pool2d(x, x.size(2))
        w = F.relu(self.fc1(w))
        w = self.fc2(w)
        w = self.gs(w, temp=temperature, force_hard=True)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.shortcut(x)+out * w[:,1].unsqueeze(1)
        out = F.relu(out)
        return out,w[:,1]

class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1,expansion=1):
        super(Bottleneck, self).__init__()
        self.expansion=expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out = self.shortcut(x)+out
        out = F.relu(out)
        return out

class GroupUnit(nn.Module):
    def __init__(self,in_planes,planes,stride=1,g=4,expansion=1):
        super(GroupUnit, self).__init__()
        #c_tag=0.5->g=2,c_tag=0.25->4
        self.expansion=expansion
        self.stride=stride
        self.group=g
        self.out_planes=self.expansion*planes
        self.g_inplanes=int(in_planes/g)
        self.g_outplanes=int(self.out_planes/g)
        self.conv1 = Bottleneck(self.g_inplanes, self.g_outplanes//4, stride=self.stride,expansion=self.expansion)
        self.conv2 = Bottleneck(self.g_inplanes, self.g_outplanes//4, stride=self.stride,expansion=self.expansion)
        self.conv3 = Bottleneck(self.g_inplanes, self.g_outplanes//4, stride=self.stride,expansion=self.expansion)
        self.conv4 = Bottleneck(self.g_inplanes, self.g_outplanes//4, stride=self.stride,expansion=self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or self.g_inplanes != self.g_outplanes//4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.g_inplanes, self.g_outplanes//4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.g_outplanes//4)
            )
            # self.shortcut = nn.Sequential(
            #     LearnedGroupConv(self.g_inplanes,self.g_outplanes,kernel_size=1,stride=stride,groups=1,condense_factor=2)
            # )
        # Gate layers
        self.fc1 = nn.Conv2d(self.g_inplanes, 16, kernel_size=1)
        self.fc2 = nn.Conv2d(16, 2, kernel_size=1)
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        self.bn=nn.BatchNorm2d(planes)

        self.gs = GumbleSoftmax()
        self.gs.cuda()

    def get_weight(self,x,temperature):
        w = F.avg_pool2d(x, x.size(2))
        w=self.fc1(w)
        w = F.relu(w)
        w = self.fc2(w)
        w = self.gs(w, temp=temperature, force_hard=True)
        return w

    def forward(self, x,temperature):
        xs=split(x,self.group)
        x1,x2,x3,x4=xs[0],xs[1],xs[2],xs[3]
        w1=self.get_weight(x1,temperature)
        w2 = self.get_weight(x2,temperature)
        w3 =self.get_weight(x3,temperature)
        w4 = self.get_weight(x4,temperature)
        w=torch.cat((w1[:,1],w2[:,1],w3[:,1],w4[:,1]),dim=1)

        out1 = self.conv1(x1)
        # out1 = (out1* w1[:, 1].unsqueeze(1)+self.shortcut(x1))+self.shortcut(x2)+ self.shortcut(x3)+ self.shortcut(x4)
        # print("out1:", (out1* w1[:, 1].unsqueeze(1)).size())
        # print("x1:", (self.shortcut(x1)).size())
        out1 = torch.cat((out1* w1[:, 1].unsqueeze(1)+ self.shortcut(x1),self.shortcut(x2), self.shortcut(x3), self.shortcut(x4)), 1)
        # out1 = torch.cat((w1[:, 1] *out1,self.shortcut(x2), self.shortcut(x3), self.shortcut(x4)), 1)
        out2=self.conv2(x2)
        out2 = torch.cat((out2* w2[:, 1].unsqueeze(1)+self.shortcut(x2), self.shortcut(x1), self.shortcut(x3), self.shortcut(x4)),1)
        # out2= (out2* w2[:, 1].unsqueeze(1)+self.shortcut(x2))+self.shortcut(x1)+ self.shortcut(x3)+self.shortcut(x4)
        # out2 = torch.cat((w2[:, 1] *out2,self.shortcut(x1), self.shortcut(x3), self.shortcut(x4)), 1)
        out3 = self.conv3(x3)
        out3 =torch.cat((out3* w3[:, 1].unsqueeze(1)+self.shortcut(x3), self.shortcut(x1), self.shortcut(x2), self.shortcut(x4)),1)
        # out3= (out3* w3[:, 1].unsqueeze(1)+self.shortcut(x3))+ self.shortcut(x1)+self.shortcut(x2)+self.shortcut(x4)
        # out3 = torch.cat((w3[:, 1] *out3,self.shortcut(x1), self.shortcut(x2), self.shortcut(x4)), 1)
        out4 = self.conv4(x4)
        out4 = torch.cat((out4* w4[:, 1].unsqueeze(1)+self.shortcut(x4), self.shortcut(x1), self.shortcut(x2), self.shortcut(x3),),1)
        # out4 = (out4* w4[:, 1].unsqueeze(1)+self.shortcut(x4))+ self.shortcut(x1)+self.shortcut(x2)+self.shortcut(x3)
        # out4 = torch.cat((w4[:, 1] *out4,self.shortcut(x1), self.shortcut(x2), self.shortcut(x3)), 1)
        out=merge([out1,out2,out3,out4])
        return out,w

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,input_size=224,expansion=1):
        super(ResNet, self).__init__()
        self.expansion=expansion
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gb=nn.AdaptiveAvgPool2d(1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3= self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 4,self.expansion))
            self.in_planes = planes * self.expansion
        return Sequential_ext(*layers)
        # return nn.Sequential(*layers)

    def forward(self, x,temperature=1, openings=None):
        gate_activations = []
        out = F.relu(self.bn1(self.conv1(x)))
        out,a = self.layer1(out,temperature)
        gate_activations.extend(a)
        out, a = self.layer2(out,temperature)
        gate_activations.extend(a)
        out, a = self.layer3(out,temperature)
        gate_activations.extend(a)
        out, a = self.layer4(out,temperature)
        gate_activations.extend(a)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out,gate_activations

def ResNet101(num_classes=10):
    return ResNet(GroupUnit, [3,4,23,3],num_classes=num_classes)


def test():
    net = ResNet101(num_classes=10)
    x = torch.randn(12, 3, 32, 32)
    y, w = net(Variable(x))
    print(y.size())
# test()