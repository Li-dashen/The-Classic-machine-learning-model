import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn
import time
import pandas as pd
import copy

def trian_val_data_process(): #数据加载
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    train_data,val_data = Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))]) #划分训练和测试

    train_data_loarder = Data.DataLoader(dataset = train_data,
                                         batch_size=128,
                                         shuffle=True,
                                         num_workers=8) #训练集

    val_data_loarder = Data.DataLoader(dataset=val_data,
                                         batch_size=128,
                                         shuffle=True,
                                        num_workers=8)  # 测试集

    return train_data_loarder,val_data_loarder

def train_model_process(model,train_dataloader,val_dataloader,num_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001) #使用Adam优化器
    criterion = nn.CrossEntropyLoss() #交叉损失函数
    model = model.to(device)#将模型放入训练设备中
    best_modelwts = copy.deepcopy(model.state_dict()) #复制当前模型参数

    #初始化参数
    #最高准度
    best_acc = 0.0
    #训练集损失函数列表
    train_loss_all = []
    #测试集损失函数列表
    val_loss_all = []
    # 训练集精度函数列表
    train_acc_all = []
    # 测试集精度函数列表
    val_acc_all = []
    #当前时间
    since = time.time()

    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch,num_epoch-1))
        print("-"*10)

        train_loss = 0.0#训练集损失函数
        train_corrects = 0.0#训练集准度
        val_loss = 0.0#测试集损失函数
        val_corrects = 0.0#测试集准度
        train_num = 0#训练集样本数量
        val_num = 0  # 测试集样本数量

        #对每一个mini_batch进行训练计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device) #将特征放入训练设备中
            b_y = b_y.to(device) #将标签放入训练设备中
            model.train()#设置为训练模式

            #前向传播，输入一个batch，输出一个batch中对应预测
            output = model(b_x)
            #查找每一行值中最大值对应的标签
            pre_lab = torch.argmax(output,dim=1)
            #计算每一个batch的损失函数
            loss = criterion(output,b_y)

            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播计算
            loss.backward()
            #根据网络反向传播梯度信息更新参数，起到降低loss函数计算值的作用
            optimizer.step()
            #对损失函数累加
            train_loss += loss.item()*b_x.size(0)
            #若预测正确，则准确度+1
            train_corrects += torch.sum(pre_lab == b_y.data)
            #当权用于训练的样本数
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)  # 将特征放入测试设备中
            b_y = b_y.to(device)  # 将标签放入测试设备中
            model.eval()  # 设置为测试模式

            # 前向传播，输入一个batch，输出一个batch中对应预测
            output = model(b_x)
            # 查找每一行值中最大值对应的标签
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 对损失函数累加
            val_loss += loss.item() * b_x.size(0)
            # 若预测正确，则准确度+1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当权用于测试的样本数
            val_num += b_x.size(0)

        #计算并保存每一次计算的loss和准确率
        # 计算并保存训练集的loss和准确率
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        # 计算并保存测试集的loss和准确率
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} val Acc: {:.4f}'.format(epoch,val_loss_all[-1], val_acc_all[-1]))

        #寻找最高精度的权重
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            #保留当前最高准确度
            best_modelwts = copy.deepcopy(model.state_dict())

        #计算耗费实践
        time_user = time.time() - since
        print("训练和验证耗费耗费时间{:.0f}m{:.0f}s".format(time_user//60,time_user%60))

    #加载最高准确率下的模型参数
    torch.save(best_modelwts,'D:/桌面/pytorch_testing/venv/LeNet5/best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epoch),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all,
                                       })
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label= "train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bo-', label= "val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.plot(train_process["epoch"], train_process.train_acc_all, 'gx-', label= "train loss")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'yx-', label= "val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__=="__main__":
    #将模型实例化
    LeNet = LeNet()
    train_dataloader, val_dataloader = trian_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)





