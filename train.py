import torch
import torch.optim as optim
from renderer import Renderer
import json

def train_model(model, train_loader, config):
    # 将模型移动到设备上（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    # 初始化渲染器，并将 rx_position 移动到设备上
    renderer = Renderer(network_fn=model)
    renderer.rx_position = renderer.rx_position.to(device)  # 确保 rx_position 在正确的设备上

    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_train_loss = 0.0

        for tx_position, targets in train_loader:
            optimizer.zero_grad()

            # 将 tx_position 和 targets 移动到设备上
            tx_position = tx_position.to(device)  # [batch_size, 2]
            targets = targets.to(device)  # [batch_size, 2]

            # 渲染得到预测的幅度和角度
            predicted_abs, predicted_angles = renderer.render(tx_position)

            # 计算损失
            target_abs = targets[:, 0]  # [batch_size]
            target_angles = targets[:, 1]  # [batch_size]
            loss = renderer.compute_loss(predicted_abs, predicted_angles, target_abs, target_angles)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        print(f"Epoch {epoch + 1}, Train Loss: {total_train_loss / len(train_loader):.4f}")

def eval_model(model, test_loader):
    """
    评估模型性能，计算预测的幅度与真实值之间的绝对值之差，并保存结果。

    Args:
        model: 已训练的模型实例。
        test_loader: 测试数据的 DataLoader 实例。
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 初始化渲染器，并将 rx_position 移动到设备上
    renderer = Renderer(network_fn=model)
    renderer.rx_position = renderer.rx_position.to(device)

    total_abs_difference = 0.0
    total_samples = 0
    results = []

    with torch.no_grad():
        for tx_position, targets in test_loader:
            # 将数据移动到设备上
            tx_position = tx_position.to(device)
            targets = targets.to(device)

            # 渲染得到预测的幅度和角度
            predicted_abs, predicted_angles = renderer.render(tx_position)

            # 分离目标值
            target_abs = targets[:, 0]  # [batch_size]

            # 计算绝对值之差
            abs_difference = torch.abs(predicted_abs - target_abs)

            # 累积总差异和样本数
            total_abs_difference += abs_difference.sum().item()
            total_samples += tx_position.size(0)

            # 保存每个样本的结果
            for i in range(tx_position.size(0)):
                result = {
                    'tx_position': tx_position[i].cpu().numpy().tolist(),
                    'predicted_abs': predicted_abs[i].item(),
                    'target_abs': target_abs[i].item(),
                    'abs_difference': abs_difference[i].item()
                }
                results.append(result)

    # 计算平均绝对值之差
    average_abs_difference = total_abs_difference / total_samples
    print(f"Average absolute difference in abs: {average_abs_difference:.4f}")

    # 将结果保存到本地文件
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Evaluation results saved to 'evaluation_results.json'.")
