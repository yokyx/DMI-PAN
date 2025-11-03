from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import random

"""等间隔采样  得到数据"""
class MyDataSet(Dataset):

    def __init__(self, data_list: list, label_list: list, transform=None):
        self.data_list = data_list
        self.label_list= label_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        directory = os.path.dirname(self.data_list[item])   # 读取当前视频文件的上一级路径
        target_path  = directory + '_aligned'

        video = []    #图片的tensor集合
        sample = 80  #采样图片的个数

        image_files = [f for f in os.listdir(target_path)]
        interval = len(image_files) // sample  # 计算间隔
        sample_files = image_files[:sample * interval:interval]         #保我们只选择 60 个元素，即使 length 不能被 interval 整除。

        for file in sample_files:
            image_path = os.path.join(target_path, file)
            image = Image.open(image_path)
            tensor = self.transform(image)
            video.append(tensor)

        video = torch.stack(video, dim=1)
        #将[3,100,224,224]转化为[100,3,224,224]，MIPA需要
        video = video.permute(1, 0, 2, 3)
        label = self.label_list[item]

        return video , label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        videos, labels = tuple(zip(*batch))

        videos = torch.stack(videos, dim=0)
        labels = torch.as_tensor(labels)
        return videos, labels
