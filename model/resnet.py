import torch
import torch.nn as nn
import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        # 第二层卷积层
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        # 第三层卷积层
        self.conv3 = nn.Conv2d(outplanes, outplanes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes * 4)

        self.relu = nn.ReLU(inplace=True)

        #连接层,继承给定好的连接层
        self.downsample = downsample


    def forward(self, x):
        # 保存初始状态x
        residual = x

        # 节点1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 节点2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 节点3
        out = self.conv3(out)
        out = self.bn3(out)

        #设置一个手动安排是否连接
        if self.downsample is not None:
            residual = self.downsample(x)

        #与初始状态进行连接
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, img_size=224,dropout_num = 0.5):
        super(ResNet, self).__init__()
        #设置节点数
        self.inplanes = 64
        #设置drop
        self.dropout_num= dropout_num

        #创建预处理层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        #创建layer1
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 创建layer2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # 创建layer3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # 创建layer4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #池化,比例为32倍数
        pool_kernel = int(img_size / 32)
        self.avgpool = nn.AvgPool2d(pool_kernel, stride=1, ceil_mode=True)

        #丢弃层
        self.dropout = nn.Dropout(self.dropout_num)

        #创建全连接层
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        #处理连接层
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        #创建多层Block--->[]
        #由于retnet50是[3,4,6,3],所以这里用for来写,后面直接给[3,4,6,3]就创建了多层block
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    #前向传播
    def forward(self, x):
        #预处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)  # Layer1

        x = self.layer2(x)  # Layer2

        x = self.layer3(x)  # Layer3

        x = self.layer4(x)  # Layer4

        #池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)#pytorch特性,要进行全连接的话,必须先展开

        x = self.dropout(x)

        #全连接
        x = self.fc(x)

        return x

def resnet50( **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model



if __name__ == "__main__":
    input = torch.randn([10, 3, 224,224])
    model = resnet50(num_classes=3, img_size=224)
    output = model(input)
    print(output.size())
    print(output)
