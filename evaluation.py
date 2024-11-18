import torch
import numpy as np
import numpy
#定义准确率计算公式
def accuracy(output, target):
    acc = 0.
    if len(output)>1:
        for i in range(len(output)):
            out = torch.sigmoid(output[i]).view(-1).data.cpu().numpy()
            out = (np.round(out)).astype('int')
            label = target[i].view(-1).data.cpu().numpy()
            label = (np.round(label)).astype('int')
            acc += (out == label).sum() / len(out)
    else:
        out = torch.sigmoid(output).view(-1).data.cpu().numpy()
        out = (np.round(out)).astype('int')
        label = target.view(-1).data.cpu().numpy()
        label = (np.round(label)).astype('int')
        acc = (out == label).sum()/len(out)
    return acc
#定义IOU系数计算公式
def iou_score(output, target): # 用
    smooth = 1e-5
    output = torch.sigmoid(output)
    #if torch.is_tensor(output):
     #   output = torch.sigmoid(output).data.cpu().numpy()
    #if torch.is_tensor(target):
     #   target = target.data.cpu().numpy()
    output_ = output > 0.5 #根据0.5判定矩阵中的元素为true或false
    target_ = target > 0.5 #根据0.5判定矩阵中的元素为true或false
    # intersection = (output_ & target_).sum()
    # union = (output_ | target_).sum()
    iou = 0.
    if len(output)>1:
        for i in range(len(output)):
            union = (output_[i] | target_[i]).sum()
            intersection = (output_[i] & target_[i]).sum()
            iou += (intersection + smooth) / (union + smooth)
    else:
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
    return iou

#定义Dice系数计算公式
def dice_coef(output, target):  # dice=2*TP/(TP+FN)+(TP+FP) 用
    smooth = 1e-5
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output=torch.sigmoid(output)
    output[output > 0.5] = 1  #将预测概率得分>0.5的输出变为与标签相匹配的值1
    output[output <= 0.5] = 0 #将预测概率得分<=0.5的输出变为与标签相匹配的值0
    # target = target.view(-1).data.cpu().numpy()
    #target = target.data.cpu().numpy()
    dice=0.
    # ipdb.set_trace()
    # '''wfy:batch_num>1时，改进'''
    if len(output)>1:
        for i in range(len(output)):
            intersection = (output[i] * target[i]).sum()
            dice += (2. * intersection + smooth)/(output[i].sum() + target[i].sum() + smooth)
    else:
        intersection = (output * target).sum() # 一个数字,=TP
        dice = (2. * intersection + smooth) /(output.sum() + target.sum() + smooth)
    # intersection = (output * target).sum()
    # return (2. * intersection + smooth) / \
    #     (output.sum() + target.sum() + smooth)
    #原本只用上面这一句
    return dice


#定义Dice系数
def get_DC(output, gt):
    output = torch.sigmoid(output)
    output = output > 0.5
    gt = gt > 0.5
    inter = torch.sum((output.byte() + gt.byte()) == 2)
    dc = float(2*inter)/(float(torch.sum(output) + torch.sum(gt)) + 1e-6)
    return dc
#定义杰卡德系数，IOU
def get_JS(output, gt):
    output = torch.sigmoid(output)
    output = output > 0.5
    gt = gt > 0.5
    inter = torch.sum((output.byte() + gt.byte()) == 2)
    union = torch.sum((output.byte() + gt.byte()) >= 1)
    js = float(inter) / (float(union) + 1e-6)
    return js
#定义敏感性=recall
def get_sensitivity(output, gt): # 求敏感度 se=TP/(TP+FN) 用recall

    SE = 0.
    output = output > 0.5
    gt = gt > 0.5
    TP = ((output==1).byte() + (gt==1).byte()) == 2
    FN = ((output==0).byte() + (gt==1).byte()) == 2
    #wfy:batch_num>1时，改进
    if len(output)>1:
        for i in range(len(output)):
            SE += float(torch.sum(TP[i])) / (float(torch.sum(TP[i]+FN[i])) + 1e-6)
    else:
        SE = float(torch.sum(TP)) / (float(torch.sum(TP+FN)) + 1e-6) #原本只用这一句
    #SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)  # 原本只用这一句
    return SE  #返回batch中所有样本的SE和
#定义特异度
def get_specificity(SR, GT, threshold=0.5):#求特异性 sp=TN/(FP+TN) 用
    SR = SR > threshold #得到true和false
    GT = GT > threshold
    SP=0.# wfy
    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    #wfy:batch_num>1时，改进
    if len(SR)>1:
        for i in range(len(SR)):
            SP += float(torch.sum(TN[i])) / (float(torch.sum(TN[i] + FP[i])) + 1e-6)
    else:
        SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6) # 原本只用这一句
    #
    # SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    return SP
#定义精确率
def get_precision(output, gt): # ppv == precision = pr = TP/(TP+FP)
    smooth = 1e-5
    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(gt):
    #     gt = gt.data.cpu().numpy()
    output = output > 0.5
    gt = gt > 0.5
    TP = ((output==1).byte() + (gt==1).byte()) == 2
    FP = ((output == 1).byte() + (gt == 0).byte()) == 2
    ppv=0.
    if len(output)>1:
        for i in range(len(output)):
            ppv += (float(torch.sum(TP[i])) + smooth)/(float(torch.sum(TP[i])) + float(torch.sum(FP[i])) + smooth)
    else:
        ppv = (float(torch.sum(TP)) + smooth)/(float(torch.sum(TP)) + float(torch.sum(FP)) + smooth)

    # intersection = (output * target).sum() # TP
    return ppv

#定义F1得分=dice
def get_F1(output, gt):
    f1 = 0.
    if len(output)>1:
        for i in range(len(output)):
            se = get_sensitivity(output[i], gt[i])
            pc = get_precision(output[i], gt[i])
            f1 += 2 * se * pc / (se + pc + 1e-6)
    else:
        se = get_sensitivity(output, gt)
        pc = get_precision(output, gt)
        f1 = 2*se*pc / (se+pc+1e-6)
    return f1