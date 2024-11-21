import torch
import torch.nn.functional as F
import numpy as np

class Renderer:
    def __init__(self, network_fn, rx_position = torch.tensor([21.588, 10.877], dtype=torch.float32), near=0.0, far=20.0, n_samples=36, num_directions=360):
        """
        初始化渲染器，用于计算信号的幅度（abs）和相位角度（angle）。

        Args:
            network_fn: NeRF模型，输入为发送端位置和相对于接收端的方向。
                        需要具备一个方法 `initialize_rx_position()` 返回接收端位置。
            near: 射线的近裁剪距离（米）。
            far: 射线的远裁剪距离（米），覆盖整体20x20米区域。
            n_samples: 每条射线的采样点数（设为36个点）。
            num_directions: 水平面上射线的数量（设为360个方向）。
        """
        self.network_fn = network_fn
        # 使用 network_fn 初始化接收端位置
        self.rx_position = rx_position
        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.num_directions = num_directions

    def sample_points(self, rays_o, rays_d):
        """
        沿射线采样点。

        Args:
            rays_o: 射线的起点，形状为 [batch_size, num_rays, 2]。
            rays_d: 射线的方向，形状为 [batch_size, num_rays, 2]。

        Returns:
            pts: 采样点，形状为 [batch_size, num_rays, n_samples, 2]。
            t_vals: 采样距离，形状为 [batch_size, num_rays, n_samples]。
        """
        batch_size, num_rays, _ = rays_o.shape

        # 生成从 near 到 far 的均匀采样距离
        t_vals = torch.linspace(self.near, self.far, self.n_samples).to(rays_o.device)  # [n_samples]
        t_vals = t_vals.expand(batch_size, num_rays, self.n_samples)  # [batch_size, num_rays, n_samples]

        # 计算采样点坐标
        rays_o = rays_o.unsqueeze(2)  # [batch_size, num_rays, 1, 2]
        rays_d = rays_d.unsqueeze(2)  # [batch_size, num_rays, 1, 2]
        t_vals_expanded = t_vals.unsqueeze(3)  # [batch_size, num_rays, n_samples, 1]

        pts = rays_o + rays_d * t_vals_expanded  # [batch_size, num_rays, n_samples, 2]
        return pts, t_vals

    def render(self, tx_positions):
        batch_size = tx_positions.shape[0]
        device = tx_positions.device
        num_rays = self.num_directions

        # 生成方向
        angles = torch.linspace(0, 2 * np.pi, self.num_directions + 1)[:-1].to(device)
        dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        dirs = dirs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_rays, 2]

        # 接收端位置
        rays_o = self.rx_position.to(device).unsqueeze(0).unsqueeze(1).expand(batch_size, num_rays, -1)  # [batch_size, num_rays, 2]

        # 采样点
        pts, _ = self.sample_points(rays_o, dirs)  # [batch_size, num_rays, n_samples, 2]

        # 从发送端到采样点的方向
        tx_positions_expanded = tx_positions.unsqueeze(1).unsqueeze(2).expand(-1, num_rays, self.n_samples, -1)  # [batch_size, num_rays, n_samples, 2]
        directions = pts - tx_positions_expanded  # [batch_size, num_rays, n_samples, 2]
        directions = F.normalize(directions, dim=-1)

        # 调整形状
        batch_size_new = batch_size * num_rays
        tx_positions_input = tx_positions_expanded.reshape(batch_size_new, self.n_samples, 2)
        directions_input = directions.view(batch_size_new, self.n_samples, 2)

        # 调用网络
        amp, phase, _ = self.network_fn(tx_positions_input, directions_input)

        # 恢复形状
        amp = amp.view(batch_size, num_rays, self.n_samples)
        phase = phase.view(batch_size, num_rays, self.n_samples)

        # 计算信号
        signal = amp * torch.exp(1j * phase)  # [batch_size, num_rays, n_samples]

        # 累积信号
        cumulative_signal_per_ray = torch.sum(signal, dim=2)  # [batch_size, num_rays]
        cumulative_signal = torch.sum(cumulative_signal_per_ray, dim=1)  # [batch_size]

        # 预测结果
        predicted_abs = torch.abs(cumulative_signal)
        predicted_angles = torch.angle(cumulative_signal)

        return predicted_abs, predicted_angles


    def compute_loss(self, predicted_abs, predicted_angles, target_abs, target_angles):
        """
        计算训练损失。

        Args:
            predicted_abs: 预测的幅度，形状为 [batch_size]。
            predicted_angles: 预测的角度，形状为 [batch_size]。
            target_abs: 目标幅度，形状为 [batch_size]。
            target_angles: 目标角度，形状为 [batch_size]。

        Returns:
            loss: 损失值。
        """
        abs_loss = F.mse_loss(predicted_abs, target_abs)
        # angle_loss = F.mse_loss(predicted_angles, target_angles)
        return abs_loss