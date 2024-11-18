from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image
from evaluation import *
from cswinmodeltiny import *
from PIL import Image
import torch
import torch.nn as nn
import torch as nn
import torch.nn.functional as F
import cv2
class Mydataset(Dataset): #获取文件夹里的图片，并返回图片和label

    def __init__(self,root_dir,label_dir,transform=None,ltransform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir #
        self.imagepath = os.path.join(self.root_dir)
        self.labpath=os.path.join(self.label_dir)
        self.imgs_path = os.listdir(self.imagepath)
        self.lab_path=os.listdir(self.labpath)
        self.transform = transform
        self.ltransform=ltransform

    def __getitem__(self, idx):
        img_name = self.imgs_path[idx]
        img_image_path = os.path.join(self.root_dir,img_name)
        lab_name=self.lab_path[idx]
        lab_label_path=os.path.join(self.label_dir,lab_name)

        img=Image.open(img_image_path).convert('RGB')
        lab=Image.open(lab_label_path).convert('1')
        
        if self.transform is not None:
            img=self.transform(img)
        if self.ltransform is not None:
            lab=self.ltransform(lab)
        #lab[lab>=127]=255
        #lab[lab<127]=0
        return img,lab

    def __len__(self):
        return len(self.imgs_path)


trainimage= "/home/zyw/paper6/cvcdb/image/"
trainlab = "/home/zyw/paper6/cvcdb/lab/"
#testimage='/hpcfiles/users/zhaoyawu/paper4/cvcdb/imagetest1/'
#testlab='/hpcfiles/users/zhaoyawu/paper4/cvcdb/labtest/'

imagetransform =  transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5 ],std=[0.5,0.5,0.5 ])])
labtransform=transforms.Compose(
    [
        #transforms.ToPILImage(), 
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]
)
data = Mydataset(trainimage, trainlab,transform=imagetransform,ltransform=labtransform)
#data2 = Mydataset(testimage, testlab,transform=imagetransform,ltransform=labtransform)
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size],torch.manual_seed(100))

train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True,drop_last=True,num_workers=4)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,drop_last=True,num_workers=4)

#img_file='/hpcfiles/users/zhaoyawu/paper4/cvcdb/attunet/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model= CSWINU_Net(3,1).to(device)
#model=NestedUNet(nInputChannels=3, n_classes=1, os=16).to(device)#inchannels,class
num_epochs=100
learning_rate = 0.0001 # 学习率
#momentum = 0.5
#momentum=0.5,weight_decay=1e-5
#定义损失函数和优化器
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.00001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.99)#固定步长衰减
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.8, patience=15,verbose=False, threshold=0.0001,min_lr=0)#固定步长衰减
criterion= BCEDiceLoss()#BCEDiceLoss 
best=0.
besti=0
for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 50)

        #执行学习率指数衰减
        length=0
        epoch_loss = 0
        step = 0
        acc=0
        SE=0
        SP=0
        PC=0
        F1=0
        JS=0
        DC=0
        for i, (images,GT) in enumerate (train_loader):
          
            images = images.to(device)
            GT = GT.to(device)
           
            #print(GT)
            #c=GT.data.cpu().numpy()
            #print(np.unique(c))
            # zero the parameter gradients
            optimizer.zero_grad()#梯度清理
            # forward
            SR =model(images)
            #SR_probs=torch.sigmoid(SR)
            #SR_flat = SR_probs.view(SR_probs.size(0), -1)
            #GT_flat = GT.view(GT.size(0), -1)
            loss = criterion(SR,GT)
            loss.backward()#误差反向传播
            optimizer.step()#基于当前更新梯度
            epoch_loss += loss.item()
            #SR_probs[SR_probs >= 0.5] = 1
            #SR_probs[SR_probs < 0.5] = 0
            #print(np.unique(SR))
            acc += accuracy(SR, GT)#准确率
            SE += get_sensitivity(SR, GT)#recall
            SP += get_specificity(SR, GT)#特异性
            PC += get_precision(SR, GT)#精确率
            F1 += get_F1(SR, GT)#F1得分
            JS += iou_score(SR, GT)
            DC += dice_coef(SR,GT)
            length += images.size(0)
        scheduler.step(loss)
        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        #IOU=IOU / length
        print(
            'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                epoch + 1, num_epochs, \
                epoch_loss, \
                acc, SE, SP, PC, F1, JS, DC))
        with torch.no_grad():
            model.eval()
            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            length = 0
            #for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
                # print(batch_idx)
             #   if isinstance(rest[0][0], str):
                    #image_filename = rest[0][0]
              #  else:
               #     image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
            for i, (images, GT,) in enumerate(test_loader):
              
                images = images.to(device)
                GT = GT.to(device)
                #print(GT)
                #c=GT.data.cpu().numpy()
                #print(np.unique(c))
                SR = model(images)
                #print(SR)
                #c=SR.data.cpu().numpy()
                #print(np.unique(c))
                b = torch.sigmoid(SR)
                #SR_probs[SR_probs >= 0.5] = 1
                #SR_probs[SR_probs < 0.5] = 0
                #GT_flat = GT.view(GT.size(0), -1)
                acc += accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += iou_score(SR, GT)
                DC += dice_coef(SR,GT)
                length += images.size(0)
                #b=SR_probs.data.cpu().numpy()
                #b[b>0.5]=255
                #b[b<=0.5]=0
                #image_filename = img_file + str(epoch) +str(i) + '.png'
                #i = i + 1
                #cv2.imwrite(image_filename,b[0,0,:,:])
                #save_image(b,"/hpcfiles/users/zhaoyawu/paper4/cvcdb/xiaobounet/ images_epoch{:02d}_batch{:03d}.png".format(
                    #epoch,i),nrow=1,padding=2)
            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            JS = JS / length
            DC = DC / length
            unet_score = JS + DC
            if DC>best:
              best=DC
              besti=epoch+1
              torch.save({'epoch':besti,
            'model_state_dict':model.state_dict(),
            'otpimizer_state_dict':optimizer.state_dict()
            },'/home/zyw/paper6/isic/bestmodelWRANet.pth'

)         
            print('[test] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
            acc, SE, SP, PC, F1, JS, DC))

print('[bestepoch and bestdice] epoch:%.4d bestDC: %.4f' % (
             besti,best))
