import cv2.cv2
import torch
import numpy as np
from model.resnet import resnet50
from torchvision import transforms
# from yolov2.model.darknet19 import darknet
from model.darknet53 import darknet
from PIL import Image
from util.Draw import Draw
from matplotlib import pyplot as plt

device = torch.device('cuda')

def all_Cap_circle():
    # 自定义格式
    image_transform = transforms.Compose([

        transforms.Resize([IMGSIZE, IMGSIZE]),  # 把图片resize为256*256

        transforms.ToTensor(),  # 将图像转为Tensor
    ])

    # 创建Model
    model = resnet50(num_classes=42, img_size=256)
    # model = resNet50(num_classes=42, img_size=256)
    model.load_state_dict(torch.load(PATH_WEIGTH_Line, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(3, IMGSIZE)
    cap.set(4, IMGSIZE)
    while True:
        # 读取摄像头图片
        flage, img = cap.read()  # bool,[y,x,c]
        pre_size = np.array(img).shape  # 获取原图大小[demo]

        # 图片初始化
        img_ = Image.fromarray(img)
        img_ = image_transform(img_).to(device)
        img_ = torch.unsqueeze(img_, dim=0)

        # 模型推理
        pred = model(img_)

        draw=Draw()
        draw.draw_circle(img, pred, per_size=pre_size, size=IMGSIZE)

        # 输入q退出
        if cv2.waitKey(1) == ord("q"):
            break

        # 打开图片
        cv2.imshow("Out", img)
    cap.release()


def all_Img_circle():
    # 自定义格式
    image_transform = transforms.Compose([

        transforms.Resize([IMGSIZE, IMGSIZE]),  # 把图片resize为256*256

        transforms.ToTensor(),  # 将图像转为Tensor
    ])

    # 创建Model
    model = resnet50(num_classes=42, img_size=256)
    # model = resNet50(num_classes=42, img_size=256)
    model.load_state_dict(torch.load(PATH_WEIGTH_Line, map_location=device))
    model.to(device)
    model.eval()

    while True:
        # 获取图像信息
        img = cv2.imread(PATH_IMG)
        pre_size = np.array(img).shape  # 获取原图大小[demo]
        # img=cv2.resize(img,(IMGSIZE,IMGSIZE))

        # 图片初始化
        img_ = Image.fromarray(img)
        img_ = image_transform(img_).to(device)
        img_ = torch.unsqueeze(img_, dim=0)

        # 模型推理
        pred = model(img_)
        draw=Draw()
        draw.draw_circle(img, pred, per_size=pre_size, size=IMGSIZE)

        # 输入q退出
        if cv2.waitKey(1) == ord("q"):
            break

        # 打开图片
        cv2.imshow("Out", img)


def all_Cap_line():
    # 自定义格式
    image_transform = transforms.Compose([

        transforms.Resize([IMGSIZE, IMGSIZE]),  # 把图片resize为256*256

        transforms.ToTensor(),  # 将图像转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 创建Model
    model = resnet50(num_classes=42, img_size=256)
    model.load_state_dict(torch.load(PATH_WEIGTH_Line, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(3, IMGSIZE)
    cap.set(4, IMGSIZE)
    while True:
        # 读取摄像头图片
        flage, img = cap.read()  # bool,[y,x,c]
        img=cv2.flip(img, 1)
        #img=img[..., ::-1].copy()
        pre_size = np.array(img).shape  # 获取原图大小[demo]
        print(pre_size)
        # 图片初始化
        img_ = Image.fromarray(img)
        img_ = image_transform(img_).to(device)
        img_ = torch.unsqueeze(img_, dim=0)

        # 模型推理
        # print("img_2",img_)
        pred = model(img_)

        draw=Draw()
        draw.draw_line(img, pred, per_size=pre_size)

        # 输入q退出
        if cv2.waitKey(1) == ord("q"):
            break

        # 打开图片
        cv2.imshow("Out", img)
    cap.release()


def yolo_Cap_rectangle():
    # 创建Model
    model = darknet()
    model.load_state_dict(torch.load(PATH_WEIGTH_YOLO, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    # 自定义格式
    image_transform = transforms.Compose([

        transforms.Resize([360, 480]),  # 把图片resize为256*256

        transforms.ToTensor(),  # 将图像转为Tensor
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    state=True
    while True:
        # 读取摄像头图片
        flage, img = cap.read()  # bool,[y,x,c]
        if state:
            print("摄像头状态:", flage)
            state=False
        img = cv2.flip(img, 1)




        # 图片初始化
        img_ = img[..., ::-1].copy()
        img_ = Image.fromarray(img_)

        img_ = image_transform(img_).to(device)
        img_ = torch.unsqueeze(img_, dim=0)
        # print(img_.shape,"img_")
        # print(img.shape,"img")
        # 模型推理
        with torch.no_grad():
            pred = model(img_)

        draw=Draw()
        draw.draw_rectangle(img, pred)

        # 输入q退出
        if cv2.waitKey(2) == ord("q"):
            break

    cap.release()


def yolo_Img_rectangle():
    # 创建Model
    model = darknet()
    model.load_state_dict(torch.load(PATH_WEIGTH_YOLO, map_location=device))
    model.to(device)
    model.eval()

    # 自定义格式
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转为Tensor
    ])
    while True:
        # 获取图像信息
        img = cv2.imread(PATH_IMG)

        # 图片初始化
        img_ = img[..., ::-1].copy()
        img_ = image_transform(img_).to(device)
        img_ = torch.unsqueeze(img_, dim=0)

        # 模型推理
        pred = model(img_)
        # print(pred, "pred")
        draw=Draw()
        draw.draw_rectangle(img, pred)

        # 输入q退出
        if cv2.waitKey(1) == ord("q"):
            break

        # 打开图片
        cv2.imshow("Out", img)


def hand_line_detect():
    # 创建Model
    model_yolo = darknet()
    model_yolo.load_state_dict(torch.load(PATH_WEIGTH_YOLO, map_location=device))
    model_yolo.to(device)
    model_yolo.eval()

    model_line = resnet50(num_classes=42, img_size=256)
    model_line.load_state_dict(torch.load(PATH_WEIGTH_Line, map_location=device))
    model_line.to(device)
    model_line.eval()

    cap = cv2.VideoCapture(0)
    # 自定义格式
    image_transform = transforms.Compose([

        transforms.Resize([360, 480]),  # 把图片resize为256*256

        transforms.ToTensor()  # 将图像转为Tensor
    ])

    image_transform_line = transforms.Compose([

        transforms.Resize([256, 256]),  # 把图片resize为256*256

        transforms.ToTensor(),  # 将图像转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    i = 0
    state=True
    while True:
        print(i)
        # 读取摄像头图片
        flage, img = cap.read()  # bool,[y,x,c]
        img = cv2.flip(img, 1)
        if state:
            print("摄像头状态:", flage)
            state = False


        # 图片初始化
        img_ = img[..., ::-1].copy()
        img_ = Image.fromarray(img_)

        img_ = image_transform(img_).to(device)
        img_ = torch.unsqueeze(img_, dim=0)
        # yolo模型推理
        with torch.no_grad():
            pred_yolo = model_yolo(img_)

        draw=Draw()
        draw.draw_hand(img, pred_yolo, model_line, image_transform_line)
        i = i + 1

        # 输入q退出
        if cv2.waitKey(2) == ord("q"):
            break

        # 打开图片
        cv2.imshow("Out", img)
    cap.release()


def Pose_detect():
    # 创建Model
    model_pose = resnet50(num_classes=3, img_size=224)
    #path= 'data/Pose_data/right/2021-05-24_1649060.jpg'
    path = 'WIN_20210524_01_03_31_Pro.jpg'
    model_pose.load_state_dict(torch.load(PATH_WRIGTH_Pose, map_location=device))
    model_pose.to(device)
    model_pose.eval()

    image_transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转为Tensor
        transforms.Normalize((0.5086418, 0.45891127, 0.4061756), (0.2057245, 0.20411159, 0.22393352))
    ])

    img = cv2.imread(path)

    # 图片初始化
    img_ = img[..., ::-1].copy()
    img_ = image_transform(img_).to(device)
    img_ = torch.unsqueeze(img_, dim=0)

    # 模型推理
    pred = model_pose(img_)
    print(torch.softmax(pred,dim=-1))
    pred = torch.argmax(pred)

    print(data_mouldes[pred.item()])

def yolo_Img_rectangle():
    # 创建Model
    model = darknet()
    model.load_state_dict(torch.load(PATH_WEIGTH_YOLO, map_location=device))
    model.to(device)
    model.eval()

    # 自定义格式
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转为Tensor
    ])
    while True:
        # 获取图像信息
        img = cv2.imread(PATH_IMG)

        pre_size = np.array(img).shape  # 获取原图大小[demo]

        # 图片初始化
        img_ = img[..., ::-1].copy()
        img_ = image_transform(img_).to(device)
        img_ = torch.unsqueeze(img_, dim=0)

        # 模型推理
        pred = model(img_)
        # print(pred, "pred")
        draw=Draw()
        draw.draw_rectangle(img, pred)

        # 输入q退出
        if cv2.waitKey(1) == ord("q"):
            break

        # 打开图片
        cv2.imshow("Out", img)


def hand_pose_detect():
    # 创建Model
    model_yolo = darknet()
    model_yolo.load_state_dict(torch.load(PATH_WEIGTH_YOLO, map_location=device))
    model_yolo.to(device)
    model_yolo.eval()

    model_pose = resnet50(num_classes=3, img_size=224)
    model_pose.load_state_dict(torch.load(PATH_WRIGTH_Pose, map_location=device))
    model_pose.to(device)
    model_pose.eval()

    cap = cv2.VideoCapture(0)
    # 自定义格式
    image_transform = transforms.Compose([

        transforms.Resize([360, 480]),  # 把图片resize为256*256

        transforms.ToTensor(),  # 将图像转为Tensor
        transforms.Normalize((0.4657646, 0.43675116, 0.38411704), (0.28656653, 0.2745257, 0.27989933)),
    ])

    image_transform_pose = transforms.Compose([

        transforms.Resize([224, 224]),  # 把图片resize为256*256

        transforms.ToTensor(),  # 将图像转为Tensor
    ])
    i = 0
    state=True
    while True:
        # 读取摄像头图片
        flage, img = cap.read()  # bool,[y,x,c]
        img = cv2.flip(img, 1)
        if state:
            print("摄像头状态:", flage)
            state = False


        # 图片初始化
        img_ = img[..., ::-1].copy()
        img_ = Image.fromarray(img_)

        img_ = image_transform(img_).to(device)
        img_ = torch.unsqueeze(img_, dim=0)
        # yolo模型推理
        with torch.no_grad():
            pred_yolo = model_yolo(img_)

        draw=Draw()
        draw.draw_pose_hand(img, pred_yolo, model_pose, image_transform_pose,data_mouldes)
        i = i + 1

        # 输入q退出
        if cv2.waitKey(2) == ord("q"):
            break

        # 打开图片
        cv2.imshow("Out", img)
    cap.release()

if __name__ == '__main__':
    # 初始化数据
    data_mouldes = {
        0: "forward",
        1: "left",
        2: "right"
    }
    IMGSIZE = 256
    BATCH = 1
    PATH = 'data/test'
    PATH_WRIGTH_Pose='weigth/Pose_Model/Pose_Model2021-05-24_171142.pth'
    PATH_WEIGTH_Line = 'weigth/Line_Model/resnet50_2021-418.pth'
    PATH_WEIGTH_YOLO = 'weigth/Yolo_Model/hand_model_2021-05-16_181423.pth'
    PATH_IMG = 'data/me/images/2021-04-29_22305412.jpg'

    # 使用摄像头画点
    # all_Cap_circle()

    # 识别模型推理摄像头获取的图片
    all_Cap_line()

    # 识别模型推理单张图片
    # all_Img_circle()

    # Yolo模型推理摄像头获取的图片
    #yolo_Cap_rectangle()

    # Yolo模型推理单个图片
    # yolo_Img_rectangle()

    # Yolo模型及画线模型同时推理
    # hand_line_detect()

    # 手势模型推理手势
    # Pose_detect()

    # Yolo模型及画线模型同时推理
    # hand_pose_detect()
