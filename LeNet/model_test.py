import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn
import copy

def test_data_process(): #数据加载
    test_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    test_data_loader = Data.DataLoader(dataset = test_data,
                                         batch_size=128,
                                         shuffle=True,
                                         num_workers=8) #训练集


    return test_data_loader

def test_model_process(model,test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)#将模型放入训练设备中

    #初始化参数
    test_corrects = 0.0
    test_num = 0

    #只进行前向传播计算，不计算梯度，从而节省内存，加快速度
    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)  # 将特征放入测试设备中
            test_data_y = test_data_y.to(device)  # 将标签放入测试设备中
            model.eval()  # 设置为测试模式
            #输出每个样本的预测值
            output = model(test_data_x)
            #查找每一行最大值的行标
            pre_lab = torch.argmax(output,dim=1) #dim:沿着列方向查找
            #若正确，则准确数+1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)#所有测试样本进行累加

    test_acc = test_corrects.double().item() / test_num
    print("测试准确率为：",test_acc)

if __name__ == "__main__":
    #加载模型
    model = LeNet()
    model.load_state_dict(torch.load('best_model.pth'))

    #加载测试数据
    test_dataloader = test_data_process()
    #test_model_process(model,test_dataloader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #设置模型为验证模型
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)

            # 遍历批次中的每个样本
            for i in range(len(pre_lab)):
                result = pre_lab[i].item()
                label = b_y[i].item()
                print("预测值：", result, "----------", "真实值：", label)



