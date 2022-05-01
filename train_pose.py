from util.dataset.Pose.getdata import getdata
from util.loss.pose_loss import loss_cro
from model.resnet import resnet50
import torch
import time
device = torch.device('cuda')
def train(EPOCH, dataload,PATH_WEIGTH=None, LR=1e-4):

    # 创建Model
    model = resnet50(num_classes=3, img_size=224)

    # 读取数据
    if PATH_WEIGTH != None:
        print("成功读取模型:", PATH_WEIGTH)
        model.load_state_dict(torch.load(PATH_WEIGTH))
    model.to(device)
    model.train()

    # 创建学习方案
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-6)

    for epoch in range(EPOCH):
        for img,lable in dataload:

            # 正向传播
            img = img.to(device)
            lable=lable.to(device)
            pred=model(img)

            # 反向传播
            loss=loss_cro(pred,lable)
            loss.backward()

            # 更新
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()

        print("*****************************************")
        print("=========>")
        print("epoch:", epoch, "\tloss:", loss.item())

    # 训练结束,获取本地时间
    loc_time = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    torch.save(model.state_dict(), r"weigth/Pose_Model" + loc_time + ".pth")
    print(loc_time)

if __name__ == '__main__':
    EPOCH=300
    BATH=2
    Path="data/Pose_data"
    data_mouldes = {
        0: "forward",
        1: "left",
        2: "right"
    }
    # 获取数据
    dataload=getdata(BATH,Path,data_mouldes)

    train(EPOCH,dataload,LR=1e-4)
