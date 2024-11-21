import torch
from torch.utils.data import DataLoader
from dataset import NeRFDataset
from utils.map_feature_extractor import MapFeatureExtractor
from models.nerf_model import NeRFModel
from train import train_model, eval_model
import yaml
import random


def load_config(config_path):
    """
    加载配置文件
    :param config_path: 配置文件路径
    :return: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 加载配置文件
    config = load_config("config.yaml")

    # 提取配置参数
    database_path = config["data"]["database_path"]
    map_path = config["data"]["map_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config["train"]["batch_size"]

    # 提取共享地图特征
    map_extractor = MapFeatureExtractor()
    map_features = map_extractor.extract_features(map_path).to(device)
    print(f"Map features shape: {map_features.shape}")

    # 获取所有 fileID
    all_file_ids = list(range(1, 177))

    # 随机划分训练集和测试集
    random.shuffle(all_file_ids)  # 随机打乱 fileID 顺序
    split_idx = int(0.7 * len(all_file_ids))  # 70% 划分点
    train_file_ids = all_file_ids[:split_idx]  # 前 70% 作为训练集
    test_file_ids = all_file_ids[split_idx:]   # 后 30% 作为测试集

    # print(f"Train file IDs: {train_file_ids}")
    # print(f"Test file IDs: {test_file_ids}")

    # 初始化数据集和数据加载器
    train_dataset = NeRFDataset(database_path, train_file_ids)
    test_dataset = NeRFDataset(database_path, test_file_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化 NeRF 模型
    model = NeRFModel(map_features=map_features).to(device)

    # 训练模型
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        config=config
    )

    # 评估模型
    print("Evaluating model...")
    eval_model(
        model,
        test_loader
    )


if __name__ == "__main__":
    main()