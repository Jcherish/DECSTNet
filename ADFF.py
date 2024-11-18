import torch.nn as nn
import torch
class ADFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, in_ch,out_ch):
        super(ADFF, self).__init__()

        self.conv1=nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()
        self.GAP=nn.AdaptiveAvgPool2d(1)


    def forward(self, x, y):

        x=self.conv1(x)
        y=self.conv1(y)
        x1=x+y
        B=self.sigmoid(x1)

        x2=self.GAP(x1)
        x3=self.conv1(x2)
        x4=self.sigmoid(x3)
        x5=self.conv1(x4)

        x6=B*x5+x
        x7=B*x5+y

        return x6,x7

