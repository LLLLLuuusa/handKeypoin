from util.dataset.Yolo.getdata import getdata
from util.dataset.Yolo.getdata import ground_truth_generator
#from yolov2.model.darknet19 import darknet
from model.darknet53 import darknet
from util.loss.yolo_loss import yolo_loss
from util.EarlyStopping import EarlyStopping
import torch
import time
import glob
import os


device = torch.device('cuda')

def train(EPOCH, dataload, train_steps,val_steps,dataload_val, PATH_WEIGTH=None, LR=1e-5):
    # 初始化 early_stopping 对象
    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True)
    # 创建Model
    model = darknet()
    if PATH_WEIGTH!=None:
        print("成功读取模型:",PATH_WEIGTH)
        model.load_state_dict(torch.load(PATH_WEIGTH))
    model.to(device)

    # 创建学习方案
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-6)

    for epoch in range(EPOCH):
        aug_dataload = ground_truth_generator(dataload, ANCHORS)
        model.train()
        tra_loss_correct=0
        for step in range(train_steps):
            imgs, bath_detector_mast, bath_matching_gt_box, matching_classes_oh, bath_gt_boxes_grid, boxes =next(aug_dataload)
            # 将数据转化device
            imgs = imgs.to(device)
            bath_detector_mast = bath_detector_mast.to(device)
            bath_matching_gt_box = bath_matching_gt_box.to(device)
            bath_gt_boxes_grid = bath_gt_boxes_grid.to(device)
            matching_classes_oh = matching_classes_oh.to(device)


            # 正向传播file #2358:  bad zipfile offset (local header sig):
            pred = model(imgs)

            # 反向传播
            loss,sub_loss = yolo_loss(bath_detector_mast, bath_matching_gt_box, matching_classes_oh, bath_gt_boxes_grid, pred)
            loss.backward()
            tra_loss_correct=loss+tra_loss_correct
            # 更新
            optimizer.step()

            # 梯度清零
            optimizer.zero_grad()
            print(loss)

        print("*****************************************")
        print("=========>")
        print("epoch:", epoch, "\tstep:", step, "\tloss:", tra_loss_correct.item()/3460,"lr:",LR)


        aug_dataload_val = ground_truth_generator(dataload_val, ANCHORS)
        val_loss_correct = 0
        model.eval()
        for step in range(val_steps):
            imgs, bath_detector_mast, bath_matching_gt_box, matching_classes_oh, bath_gt_boxes_grid, boxes = next(
                aug_dataload_val)
            # 将数据转化device
            imgs = imgs.to(device)
            bath_detector_mast = bath_detector_mast.to(device)
            bath_matching_gt_box = bath_matching_gt_box.to(device)
            bath_gt_boxes_grid = bath_gt_boxes_grid.to(device)
            matching_classes_oh = matching_classes_oh.to(device)

            # 正向传播file #2358:  bad zipfile offset (local header sig):
            with torch.no_grad():
                pred = model(imgs)

            # 反向传播
            val_loss, val_sub_loss = yolo_loss(bath_detector_mast, bath_matching_gt_box, matching_classes_oh,
                                       bath_gt_boxes_grid, pred)
            val_loss_correct=val_loss+val_loss_correct

        valid_loss=val_loss_correct/4
        print("epoch:", epoch, "\tval_loss:", valid_loss.item())
        early_stopping(valid_loss, model)

        # 若满足 early stopping 要求
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     # 结束模型训练
        #     break
        # # 学习率优化策略
        # if early_stopping.counter == 10 or early_stopping.counter == 15:
        #     LR = LR / 10

    #if loss<0.1:
        #break

    # 训练结束,获取本地时间
    loc_time = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    print(loc_time)

    # 保存模型,不好用
    #torch.save(model, "weigth_yolo/hand_model_"+loc_time+".pth")
    # 保存模型参数
    torch.save(model.state_dict(), r"../weigth/hand_model_"+loc_time+".pth")

def test(EPOCH, dataload,steps,PATH_WEIGTH=None):
    model = darknet()
    model.train()
    # model.eval()
    # model = resNet50(num_classes=42, img_size=256)
    if PATH_WEIGTH != None:
        print("成功读取", PATH_WEIGTH)
        model.load_state_dict(torch.load(PATH_WEIGTH))
    model.to(device)

    for epoch in range(EPOCH):
        aug_dataload = ground_truth_generator(dataload, ANCHORS)
        for step in range(420):
            # 420 48
            imgs, bath_detector_mast, bath_matching_gt_box, matching_classes_oh, bath_gt_boxes_grid, boxes = next(
                aug_dataload)
            # 将数据转化device
            imgs = imgs.to(device)
            bath_detector_mast = bath_detector_mast.to(device)
            bath_matching_gt_box = bath_matching_gt_box.to(device)
            bath_gt_boxes_grid = bath_gt_boxes_grid.to(device)
            matching_classes_oh = matching_classes_oh.to(device)

            # 正向传播file #2358:  bad zipfile offset (local header sig):
            pred = model(imgs)

            # 反向传播
            loss, sub_loss = yolo_loss(bath_detector_mast, bath_matching_gt_box, matching_classes_oh,
                                       bath_gt_boxes_grid, pred)
            loss.backward()

            if step % 10 == 0:
                print("*****************************************")
                print("=========>")
                print("epoch:", epoch, "\tstep:", step, "\tloss:", loss.item())
            # print("*****************************************")
            # print("=========>")
            # print("epoch:", epoch, "\tstep:", step, "\tloss:", loss.item())

if __name__ == '__main__':
    #初始化数据
    IMGSIZE = 256
    BATCH = 8
    PATH = "data/anno"
    PATH_val = "data/me"
    PATH_WEIGTH = "weigth/Yolo_Model/hand_model_2021-05-16_181423.pth"
    ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

    train_img_files = glob.glob(os.path.join(PATH, "images/*.jpg"))
    train_steps=int(len(train_img_files)/BATCH)
    val_img_files = glob.glob(os.path.join(PATH_val, "images/*.jpg"))
    val_steps = int(len(val_img_files) / BATCH)

    # 创建dataload
    dataload = getdata(BATCH=BATCH, PATH=PATH)
    # 对dataload进行预处理
    #dataload_aug = ground_truth_generator(dataload, ANCHORS)
    dataload_val = getdata(BATCH=BATCH, PATH=PATH_val)
    train(100,dataload,train_steps,val_steps,dataload_val,PATH_WEIGTH=PATH_WEIGTH,LR=1e-6)
    #train(300, dataload, steps, dataload_val, LR=1e-5)
