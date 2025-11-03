import os
import sys
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import datetime
import torch.nn.functional as F


def setup_logging_and_dirs(args):
    if args.record:
        print("已生成logs记录实验结果")
        log_dir = "logs"
        # 检查文件夹是否存在，如果不存在，则创建
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        #步骤1: 获取实验开始时间
        start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{log_dir}/{start_time}.log"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=log_filename, filemode='w')
        #混淆矩阵图的保存路径
        ConMat_dir = "ConMat"
        if not os.path.exists(ConMat_dir):
            os.makedirs(ConMat_dir)
        dir_path = os.path.join(ConMat_dir, start_time)
        os.makedirs(dir_path)
        return dir_path
    return None

def plot_confusion_matrix(confusion_matrix,save_path):
    """
    绘制混淆矩阵图并保存到指定路径

    参数:
    - confusion_matrix: 混淆矩阵，格式为[[TP, FP], [FN, TN]]
    - save_path: 图像保存路径
    """
    # 获取真正例、假正例、假负例和真负例的数量
    TP, FP = confusion_matrix[0]
    FN, TN = confusion_matrix[1]

    # 创建混淆矩阵图
    plt.figure(figsize=(6, 4))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # 添加标签
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['1', '0'])
    plt.yticks(tick_marks, ['1', '0'])

    # 添加文本
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i][j]), horizontalalignment='center', verticalalignment='center')

    # 设置坐标轴标签
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')

    # 保存图像到指定路径
    plt.savefig(save_path)


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)             #函数用于将 Python 对象转换成二进制文件


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)            #读取给定的二进制对象数据，并将其转换为 Python 对象
        return info_list

def select_Pred_error(record_path,fold,contrast,test_path,prdicted,lable):
    #记录预测错误文件的路径，将其和混淆矩阵保存在一个文件下
    with open(f"{record_path}/{fold+ 1}.txt", 'w') as f:
        for i, flag in enumerate(contrast):
            if flag is False:
                file_name = test_path[i]
                true_label = str(lable[i])
                false_label = str(prdicted[i])
                f.write(f"{file_name}  TrueLabel:{true_label}  PredictLabel:{false_label}\n")


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.L1Loss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, ncols=120, file=sys.stdout)                 #Python 进度条库，可以在 Python 长循环中添加一个进度提示信息。
    MAE = torch.zeros(1).to(device)
    MSE = torch.zeros(1).to(device)        #Python 进度条库，可以在 Python 长循环中添加一个进度提示信息。

    for step, data in enumerate(data_loader):

        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))

        pred = pred.squeeze(-1)
        thre = [1 for x in range(0, labels.shape[0])]
        threshold = torch.as_tensor(thre).to(device)
        accu_num += torch.le(torch.abs(pred - labels.to(device)), threshold).sum()
        labels = labels.to(torch.float)

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        # 计算MAE MSE
        labels = labels.tolist()
        pred = pred.tolist()
        y_hat = torch.as_tensor(labels)
        y = torch.as_tensor(pred)
        MAE += torch.mean(torch.abs(y - y_hat).float())
        MSE += torch.mean(torch.square(y - y_hat).float())

        # .detach() 把网络中的一部分分量从反向传播的流程中拿出来，使之requires_grad=False

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, MSE: {:.3f}, MAE: {:.3f}".format(epoch,accu_loss.item() / (step + 1),
                                                                                                         accu_num.item() / sample_num,
                                                                                                         MSE.item() / (step + 1),
                                                                                                         MAE.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num , MAE.item() / (step + 1), MSE.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.L1Loss()
    model.eval()     #框架会自动把BN和DropOut固定住，不会取平均，而是用train中训练好的值

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader, ncols=120, file=sys.stdout)  # Python 进度条库，可以在 Python 长循环中添加一个进度提示信息。
    MAE = torch.zeros(1).to(device)
    MSE = torch.zeros(1).to(device)  # Python 进度条库，可以在 Python 长循环中添加一个进度提示信息。

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))

        pred = pred.squeeze(-1)
        thre = [1 for x in range(0, labels.shape[0])]
        threshold = torch.as_tensor(thre).to(device)
        accu_num += torch.le(torch.abs(pred - labels.to(device)), threshold).sum()
        labels = labels.to(torch.float)

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # 计算MAE MSE
        labels = labels.tolist()
        pred = pred.tolist()
        y_hat = torch.as_tensor(labels)
        y = torch.as_tensor(pred)
        MAE += torch.mean(torch.abs(y - y_hat).float())
        MSE += torch.mean(torch.square(y - y_hat).float())

        data_loader.desc = "[test epoch {}] loss: {:.3f}, acc: {:.3f}, MSE: {:.3f}, MAE: {:.3f}".format(epoch,accu_loss.item() / (step + 1),
                                                                                                         accu_num.item() / sample_num,
                                                                                                         MSE.item() / (step + 1),
                                                                                                         MAE.item() / (step + 1))


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, MAE.item() / (step + 1), MSE.item() / (step + 1)


#针对二分类更多指标的计算
def train_one_epoch_2(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()
    data_loader = tqdm(data_loader,ncols=150, file=sys.stdout)                 #Python 进度条库，可以在 Python 长循环中添加一个进度提示信息。

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    pred_probs = []

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))

        #保存正类的预测概率 为了后续绘制ROC曲线
        pred_prob = F.softmax(pred,dim=1)
        class1_probs = pred_prob[:, 1]
        pred_probs.extend(class1_probs.view(-1).tolist())

        pred_classes = torch.max(pred, dim=1)[1]

        # 更新TP, FP, FN, TN
        TP += ((pred_classes.to(device) == 1) & (labels.to(device) == 1)).sum().item()
        FP += ((pred_classes.to(device) == 1) & (labels.to(device) == 0)).sum().item()
        FN += ((pred_classes.to(device) == 0) & (labels.to(device) == 1)).sum().item()
        TN += ((pred_classes.to(device) == 0) & (labels.to(device) == 0)).sum().item()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        loss = loss_function(pred.to(device), labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        # .detach() 把网络中的一部分分量从反向传播的流程中拿出来，使之requires_grad=False

        data_loader.desc = "[train epoch {}] loss: {:.3f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}, accuracy: {:.3f}".format(
                                      epoch, accu_loss.item() / (step + 1), precision, recall, F1, accuracy)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), precision, recall, F1, accuracy

@torch.no_grad()
def evaluate_2(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()     #框架会自动把BN和DropOut固定住，不会取平均，而是用train中训练好的值
    data_loader = tqdm(data_loader, ncols=150, file=sys.stdout)

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    contrast = [ ]
    pred_probs = []
    pred_class = []

    for step, data in enumerate(data_loader):
        images, labels = data
        output = model(images.to(device))

        #保存正类的预测概率 为了后续绘制ROC曲线
        output_prob = F.softmax(output,dim=1)
        class1_probs = output_prob[:, 1]
        pred_probs.extend(class1_probs.view(-1).tolist())

        # 保存正类的预测类别 为了后续绘制混淆矩阵
        pred_classes = torch.max(output, dim=1)[1]
        pred_class.extend(pred_classes.view(-1).tolist())

        # 更新TP, FP, FN, TN
        TP += ((pred_classes.to(device) == 1) & (labels.to(device) == 1)).sum().item()
        FP += ((pred_classes.to(device) == 1) & (labels.to(device) == 0)).sum().item()
        FN += ((pred_classes.to(device) == 0) & (labels.to(device) == 1)).sum().item()
        TN += ((pred_classes.to(device) == 0) & (labels.to(device) == 0)).sum().item()

        # 使用 torch.eq 比较预测结果和真实标签
        cont = torch.eq(pred_classes.to(device), labels.to(device))
        contrast.extend(cont.view(-1).tolist())

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        confusion_matrix = [[TP, FP],
                            [FN, TN]]

        loss = loss_function(output.to(device), labels.to(device))
        accu_loss += loss

        data_loader.desc = "[test epoch {}] loss: {:.3f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}, accuracy: {:.3f}".format(
                                      epoch, accu_loss.item() / (step + 1), precision, recall, F1, accuracy)

    return accu_loss.item() / (step + 1), precision, recall, F1, accuracy, confusion_matrix, contrast, pred_probs, pred_class
