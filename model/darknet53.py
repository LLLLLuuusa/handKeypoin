import torch
import torch.nn as nn

class darknet(nn.Module):
    def __init__(self):
        super(darknet, self).__init__()
        #------------------------------------预处理层
        self.strm1=nn.Conv2d(3,32,kernel_size=3,padding=1,bias=False)
        self.strmbh1 = nn.BatchNorm2d(32)

        self.strm2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.strmbh2 = nn.BatchNorm2d(64)
        #------------------------------------Unit1
        self.conv1 = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # ------------------------------------Unit2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        # ------------------------------------
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        # ------------------------------------Unit3
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        # ------------------------------------
        self.conv9 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        self.bn9 = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(256)

        self.conv11 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(128)

        self.conv12 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(256)

        self.conv13 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        self.bn13 = nn.BatchNorm2d(128)

        self.conv14 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(256)

        self.conv15 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(128)

        self.conv16 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(256)

        self.conv17 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        self.bn17 = nn.BatchNorm2d(128)

        self.conv18 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(256)

        self.conv19 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        self.bn19 = nn.BatchNorm2d(128)

        self.conv20 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(256)

        self.conv21 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(128)

        self.conv22 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(256)

        self.conv23 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False)
        self.bn23 = nn.BatchNorm2d(128)

        self.conv24 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn24 = nn.BatchNorm2d(256)
        # ------------------------------------Unit4
        self.conv25 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn25 = nn.BatchNorm2d(512)
        # ------------------------------------
        self.conv26 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.bn26 = nn.BatchNorm2d(256)

        self.conv27 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn27 = nn.BatchNorm2d(512)

        self.conv28 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.bn28 = nn.BatchNorm2d(256)

        self.conv29 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(512)

        self.conv30 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.bn30 = nn.BatchNorm2d(256)

        self.conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(512)

        self.conv32 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.bn32 = nn.BatchNorm2d(256)

        self.conv33 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(512)

        self.conv34 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.bn34 = nn.BatchNorm2d(256)

        self.conv35 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn35 = nn.BatchNorm2d(512)

        self.conv36 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.bn36 = nn.BatchNorm2d(256)

        self.conv37 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn37 = nn.BatchNorm2d(512)

        self.conv38 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.bn38 = nn.BatchNorm2d(256)

        self.conv39 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn39 = nn.BatchNorm2d(512)

        self.conv40 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.bn40 = nn.BatchNorm2d(256)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(512)
        # ------------------------------------Unit5
        self.conv42 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(1024)
        # ------------------------------------
        self.conv43 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False)
        self.bn43 = nn.BatchNorm2d(512)

        self.conv44 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn44 = nn.BatchNorm2d(1024)

        self.conv45 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False)
        self.bn45 = nn.BatchNorm2d(512)

        self.conv46 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn46 = nn.BatchNorm2d(1024)

        self.conv47 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False)
        self.bn47 = nn.BatchNorm2d(512)

        self.conv48 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn48 = nn.BatchNorm2d(1024)

        self.conv49 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False)
        self.bn49 = nn.BatchNorm2d(512)

        self.conv50 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn50 = nn.BatchNorm2d(1024)

        self.conv51 = nn.Conv2d(1024+512+256 , 5 * 6, kernel_size=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.leakrelu = nn.LeakyReLU(inplace=True,negative_slope=0.1)

    def forward(self,x):
        # strm
        out = self.strm1(x)
        out = self.strmbh1(out)
        out = self.leakrelu(out)
        out = self.strm2(out)
        out = self.strmbh2(out)
        out = self.leakrelu(out)

        out = self.maxpool(out)
        # unit 1
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.leakrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakrelu(out)

        # unit 2
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.leakrelu(out)

        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.leakrelu(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.leakrelu(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.leakrelu(out)
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.leakrelu(out)
        # unit 3
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.leakrelu(out)

        out = self.maxpool(out)

        out = self.conv9(out)
        out = self.bn9(out)
        out = self.leakrelu(out)
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.leakrelu(out)
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.leakrelu(out)
        out = self.conv12(out)
        out = self.bn12(out)
        out = self.leakrelu(out)
        out = self.conv13(out)
        out = self.bn13(out)
        out = self.leakrelu(out)
        out = self.conv14(out)
        out = self.bn14(out)
        out = self.leakrelu(out)
        out = self.conv15(out)
        out = self.bn15(out)
        out = self.leakrelu(out)
        out = self.conv16(out)
        out = self.bn16(out)
        out = self.leakrelu(out)
        out = self.conv17(out)
        out = self.bn17(out)
        out = self.leakrelu(out)
        out = self.conv18(out)
        out = self.bn18(out)
        out = self.leakrelu(out)
        out = self.conv19(out)
        out = self.bn19(out)
        out = self.leakrelu(out)
        out = self.conv20(out)
        out = self.bn20(out)
        out = self.leakrelu(out)
        out = self.conv21(out)
        out = self.bn21(out)
        out = self.leakrelu(out)
        out = self.conv22(out)
        out = self.bn22(out)
        out = self.leakrelu(out)
        out = self.conv23(out)
        out = self.bn23(out)
        out = self.leakrelu(out)
        out = self.conv24(out)
        out = self.bn24(out)
        out = self.leakrelu(out)

        #残差1
        scale1 = out
        #unit 4
        out = self.conv25(out)
        out = self.bn25(out)
        out = self.leakrelu(out)

        out = self.maxpool(out)

        out = self.conv26(out)
        out = self.bn26(out)
        out = self.leakrelu(out)
        out = self.conv27(out)
        out = self.bn27(out)
        out = self.leakrelu(out)
        out = self.conv28(out)
        out = self.bn28(out)
        out = self.leakrelu(out)
        out = self.conv29(out)
        out = self.bn29(out)
        out = self.leakrelu(out)
        out = self.conv30(out)
        out = self.bn30(out)
        out = self.leakrelu(out)
        out = self.conv31(out)
        out = self.bn31(out)
        out = self.leakrelu(out)
        out = self.conv32(out)
        out = self.bn32(out)
        out = self.leakrelu(out)
        out = self.conv33(out)
        out = self.bn33(out)
        out = self.leakrelu(out)
        out = self.conv34(out)
        out = self.bn34(out)
        out = self.leakrelu(out)
        out = self.conv35(out)
        out = self.bn35(out)
        out = self.leakrelu(out)
        out = self.conv36(out)
        out = self.bn36(out)
        out = self.leakrelu(out)
        out = self.conv37(out)
        out = self.bn37(out)
        out = self.leakrelu(out)
        out = self.conv38(out)
        out = self.bn38(out)
        out = self.leakrelu(out)
        out = self.conv39(out)
        out = self.bn39(out)
        out = self.leakrelu(out)
        out = self.conv40(out)
        out = self.bn40(out)
        out = self.leakrelu(out)
        out = self.conv41(out)
        out = self.bn41(out)
        out = self.leakrelu(out)

        # 残差2
        scale2 = out
        # unit 5
        out = self.conv42(out)
        out = self.bn42(out)
        out = self.leakrelu(out)

        out = self.maxpool(out)

        out = self.conv43(out)
        out = self.bn43(out)
        out = self.leakrelu(out)
        out = self.conv44(out)
        out = self.bn44(out)
        out = self.leakrelu(out)
        out = self.conv45(out)
        out = self.bn45(out)
        out = self.leakrelu(out)
        out = self.conv46(out)
        out = self.bn46(out)
        out = self.leakrelu(out)
        out = self.conv47(out)
        out = self.bn47(out)
        out = self.leakrelu(out)
        out = self.conv48(out)
        out = self.bn48(out)
        out = self.leakrelu(out)
        out = self.conv49(out)
        out = self.bn49(out)
        out = self.leakrelu(out)
        out = self.conv50(out)
        out = self.bn50(out)
        out = self.leakrelu(out)

        #残差连接
        scale1=self.maxpool(scale1)
        scale1=self.maxpool(scale1)

        scale2=self.maxpool(scale2)
        out=torch.cat([out,scale1,scale2],dim=1)

        # unit 19
        out = self.conv51(out)
        b = out.shape[0]
        out = torch.reshape(out,[b,5,6,11,15])
        return out

if __name__ == '__main__':
    model=darknet()

    x = torch.randn([2, 3, 360, 480])
    #print(x)
    print("put", x.shape)
    output = model(x)
    print("out:",output.size())#[5,7,16,16]
