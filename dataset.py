import torch
import pandas as pd
from torch.utils.data import Dataset

class NeRFDataset(Dataset):
    def __init__(self, csv_path, file_ids):
        """
        初始化数据集
        :param csv_path: 数据集 CSV 文件路径
        :param file_ids: 需要加载的 fileID 列表
        """
        self.data = pd.read_csv(csv_path)
        # 筛选指定 fileID 的数据
        self.data = self.data[self.data['fileID'].isin(file_ids)].reset_index(drop=True)

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: 数据索引
        :return: 位置、方向、地图特征和目标值
        """
        row = self.data.iloc[idx]
        position = torch.tensor([row["x"], row["y"]], dtype=torch.float32)

        # 加载目标值
        angle = row["angle"]
        abs_value = row["abs"]
        target = torch.tensor([abs_value, angle], dtype=torch.float32)

        # 返回位置、方向、共享地图特征、目标值
        return position, target