import torch.nn as nn
import torch.nn.functional as F
import torch
from  cswinlasttiny import *
from cswintiny import *
from ADFF import *
#定义损失函数
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
    def forward(self,input,target):
        bce=F.binary_cross_entropy_with_logits(input,target)
        smooth=1e-5
        input=torch.sigmoid(input)
        num=target.size(0)
        input=input.view(num,-1)
        target=target.view(num,-1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dic = 1 - dice.sum() / num
        return 0.4 * dic+ 0.6 *bce

#开始定义U-Net网络
#step 1： 定义第一个卷积模块
class conv_block(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(conv_block, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.conv(x)
        return x

class conv_block1(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(conv_block1, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=5, stride=1,padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.conv(x)
        return x
#step 2：定义上采样模块
class up_conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(up_conv, self).__init__()
        self.up=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.up(x)
        return x

#step 3:定义U-Net模型
class CSWINU_Net(nn.Module):
    def __init__(self,in_ch=3,out_ch=1) :
        super(CSWINU_Net, self).__init__()

        n1=8
        filters=[n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16] #4个不同的卷积核数量
        #构建编码器第一分支
        self.Conv1=conv_block(in_ch, filters[0])
        self.Conv2=conv_block(filters[0], filters[1])
        self.Conv3=conv_block(filters[1], filters[2])
        self.Conv4=conv_block(filters[2], filters[3])
        self.Conv5=conv_block(filters[3], filters[4])
        #构建编码器第二分支
        self.Conv11 = conv_block1(in_ch, filters[0])
        self.Conv22 = conv_block1(filters[0], filters[1])
        self.Conv33 = conv_block1(filters[1], filters[2])
        self.Conv44 = conv_block1(filters[2], filters[3])
        self.Conv55 = conv_block1(filters[3], filters[4])
        #构建池化层
        self.Maxpool1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3=nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4=nn.MaxPool2d(kernel_size=2, stride=2)

        #注意力模块
        self.ADFF1 = ADFF(in_ch=filters[0], out_ch=filters[0])
        self.ADFF2 = ADFF(in_ch=filters[1], out_ch=filters[1])
        self.ADFF3 = ADFF(in_ch=filters[2], out_ch=filters[2])
        self.ADFF4 = ADFF(in_ch=filters[3], out_ch=filters[3])
        self.ADFF5 = ADFF(in_ch=filters[4], out_ch=filters[4])

        #构建解码器


        self.Up_conv7=conv_block(512, 512) #64---512;  32---256; 16---128; 8---64
        self.Up5 = up_conv(512,256)

        self.Up_conv6 = conv_block(384, 256)

        self.Up4=up_conv(256, 128)
        self.Up_conv5=conv_block(192, 128)

        self.Up3=up_conv(128, 64)
        self.Up_conv4=conv_block(96, 64)

        self.attention = CSWin()
        self.attention1=CSWin1()

        self.Up2=up_conv(256, 32)
        self.Up_conv2=conv_block(48, 8)

        self.Conv=nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
    def forward(self,x):  # x输入为[16,32,256,256]
        e1=self.Conv1(x)# output：[16,32,256,256]
        e11=self.Conv11(x)  # output：[16,32,256,256]

        a1,a2=self.ADFF1(e1,e11)

        e2=self.Maxpool1(a1)  # output: [16,32,128,128]
        e22=self.Maxpool1(a2) # output: [16,32,128,128]
        e2=self.Conv2(e2)  # output: [16,64,128,128]
        e22=self.Conv22(e22)  # output: [16,64,128,128]
        a3, a4 = self.ADFF2(e2, e22)

        e3=self.Maxpool2(a3) # output: [16,64,64,64]
        e33=self.Maxpool2(a4) # output: [16,64,64,64]
        e3=self.Conv3(e3)# output: [16,128,64,64]
        e33=self.Conv33(e33) # output: [16,128,64,64]
        a5, a6 = self.ADFF3(e3, e33)

        e4=self.Maxpool3(a5) # output: [16,128,32,32]
        e44=self.Maxpool3(a6) # output:[16,128,32,32]
        e4=self.Conv4(e4) # output: [16,256,32,32]
        e44=self.Conv44(e44) # output: [16,256,32,32]

        a7, a8 = self.ADFF4(e4, e44)

        e5=self.Maxpool4(a7)# output: [16,256,16,16]
        e55=self.Maxpool4(a8)  # output:[16,256,16,16]

        e5=self.Conv5(e5)# output: [4,128,16,16]
        e55=self.Conv5(e55)#output [4,128,16,16]

        a9, a10 = self.ADFF5(e5, e55)

        e6=self.attention(a9)#output [16,512,16,16]

        e66=self.attention(a10)#output [16,512,16,16]
        #插入CBAM模块
        e77=torch.cat((e6,e66),dim=1)#output [16,1024,16,16]


        d7=self.Up_conv7(e77) #output[16,512,16,16]

        d6=self.Up5(d7)#output；[16，256,32,32]
        d6 = torch.cat((e4, e44, d6), dim=1) # output: [16,384,32,32]

        d6=self.Up_conv6(d6)  #output: [16,256,32,32]


        d5=self.Up4(d6)
        d5=torch.cat((e3,e33,d5),dim=1)

        d5=self.Up_conv5(d5)#output[16,192,64,64]

        d4=self.Up3(d5)
        d4 = torch.cat((e2, e22, d4), dim=1)
        d4=self.Up_conv4(d4)#output[16,64,128,128]



        d3=self.attention1(d4)


        d2=self.Up2(d3)
        d2=torch.cat((e1,e11,d2),dim=1)

        d2=self.Up_conv2(d2)


        out=self.Conv(d2)
        return out








