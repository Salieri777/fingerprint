import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor

class MapFeatureExtractor:
    def __init__(self):
        # 使用预训练的 ResNet18 模型
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 去掉全连接层

    def extract_features(self, map_path):
        """
        提取 map.png 的全局特征。
        :param map_path: 地图图片的路径
        :return: 提取的全局特征，形状为 (1, feature_dim)
        """
        # 加载并转换图片
        map_image = Image.open(map_path).convert('RGB')  # 确保为 RGB 图像
        map_image = ToTensor()(map_image)  # 转为张量，形状 [3, H, W]
        map_image = map_image.unsqueeze(0)  # 添加批次维度，形状 [1, 3, H, W]

        # 提取特征
        features = self.feature_extractor(map_image)  # 输出形状为 [1, feature_dim, 1, 1]
        return features.squeeze()  # 返回 [1, feature_dim]