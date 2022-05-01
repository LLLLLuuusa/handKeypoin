import torch
import torch.nn as nn

class darknet(nn.Module):
    def __init__(self):
        super(darknet, self).__init__()

        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)

        self.conv10 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.bn10 = nn.BatchNorm2d(256)

        self.conv11 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)

        self.conv12 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(256)

        self.conv13 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)

        self.conv14 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(1024)

        self.conv15 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)

        self.conv16 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(1024)

        self.conv17 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False)
        self.bn17 = nn.BatchNorm2d(512)

        self.conv18 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(1024)

        self.conv19 = nn.Conv2d(1024+512, 5*6, kernel_size=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.leakrelu = nn.LeakyReLU(inplace=True,negative_slope=0.1)

    def forward(self,x):
        # unit 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakrelu(out)
        out = self.maxpool(out)

        # unit 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakrelu(out)
        out = self.maxpool(out)

        # unit 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.leakrelu(out)

        # unit 4
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.leakrelu(out)

        # unit 5
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.leakrelu(out)
        out = self.maxpool(out)

        # unit 6
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.leakrelu(out)

        # unit 7
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.leakrelu(out)

        # unit 8
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.leakrelu(out)
        out = self.maxpool(out)

        # unit 9
        out = self.conv9(out)
        out = self.bn9(out)
        out = self.leakrelu(out)

        # unit 10
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.leakrelu(out)

        # unit 11
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.leakrelu(out)

        # unit 12
        out = self.conv12(out)
        out = self.bn12(out)
        out = self.leakrelu(out)

        # unit 13
        out = self.conv13(out)
        out = self.bn13(out)
        out = self.leakrelu(out)
        out = self.maxpool(out)

        #建立残差
        skip_out = out

        # unit 14
        out = self.conv14(out)
        out = self.bn14(out)
        out = self.leakrelu(out)

        # unit 15
        out = self.conv15(out)
        out = self.bn15(out)
        out = self.leakrelu(out)

        # unit 16
        out = self.conv16(out)
        out = self.bn16(out)
        out = self.leakrelu(out)

        # unit 17
        out = self.conv17(out)
        out = self.bn17(out)
        out = self.leakrelu(out)

        # unit 18
        out = self.conv18(out)
        out = self.bn18(out)
        out = self.leakrelu(out)

        #残差连接
        out=torch.cat([out,skip_out],dim=1)

        # unit 19
        out = self.conv19(out)
        b=out.shape[0]
        out = torch.reshape(out,[b,5,6,11,15])
        return out

if __name__ == '__main__':
    model=darknet()

    x = torch.randn([2, 3, 360, 480])
    print(x)
    print("put", x.shape)
    output = model(x)
    print("out:",output.size())#[5,7,16,16]
