from util.dataset.Line.getdata import getdata
from model.resnet import resnet50
from util.loss.line_loss import *
import time

device=torch.device('cuda')
print(torch.cuda.is_available())

def train(EPOCH=None):

    #获取dataload
    dataload=getdata(IMGSIZE=IMGSIZE,BATCH=BATCH,PATH=PATH)

    #创建Model
    model = resnet50(num_classes=42, img_size=256)
    #model = resNet50(num_classes=42, img_size=256)
    model.load_state_dict(torch.load(PATH_WEIGTH))
    model.to(device)

    #创建学习方案
    optimizer=torch.optim.Adam(model.parameters(),lr=LR, betas=(0.9, 0.99),weight_decay=1e-6)

    for epoch in range(EPOCH):
        for step, (imgs, ptses) in enumerate(dataload):
            #将数据转化device
            imgs=imgs.to(device)
            true=ptses.to(device)

            #图片预处理
            imgs=(imgs-128.)/256.

            #正向传播file #2358:  bad zipfile offset (local header sig):
            pred=model(imgs)
            #print(pred)

            # 反向传播
            loss=loss_pts_wing(pred,true)
            loss.backward()

            if step%100==0:
                print("*****************************************")
                print("=========>")
                print("epoch:",epoch,"\tstep:",step,"\tloss:",loss.item())

            #更新
            optimizer.step()

            #梯度清零
            optimizer.zero_grad()

    #训练结束,获取本地时间
    loc_time = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())

    #保存模型
    #torch.save(model, "weigth/hand_model_"+loc_time+".pth")
    #保存模型参数
    #torch.save(model.state_dict(), "weigth/hand_model_weigth_"+loc_time+".pth")


if __name__ == '__main__':
    #初始化数据
    IMGSIZE=256
    BATCH=2
    PATH='data/test'
    PATH_WEIGTH='weigth/hand_model_weigth_2021-03-08_045118.pth'
    EPOCH=1
    LR=1e-3

    train(EPOCH=EPOCH)