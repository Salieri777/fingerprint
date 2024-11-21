import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.positional_encoding import get_embedder


class AttenuationNetwork(nn.Module):
    """
    衰减网络，包含 8 层全连接，第 5 层跳跃连接
    """
    def __init__(self, pos_dim, map_feature_dim, hidden_dim=256, feature_dim=256, multires=10):
        super(AttenuationNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        # 创建位置编码器
        self.embed_fn, embed_dim = get_embedder(multires, pos_dim)

        # 第一层：位置编码和地图特征拼接
        self.fc1 = nn.Linear(embed_dim + map_feature_dim, hidden_dim)

        # 中间层，包含跳跃连接
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) if i != 4 else nn.Linear(hidden_dim + embed_dim + map_feature_dim, hidden_dim)
            for i in range(7)
        ])

        # 最后一层输出特征
        self.feature_output = nn.Linear(hidden_dim, feature_dim)
        self.amp_output = nn.Linear(hidden_dim, 1)
        self.phase_output = nn.Linear(hidden_dim, 1)

    def forward(self, x, map_features):
        """
        Forward pass of the AttenuationNetwork.

        Args:
            x: [batch_size_new, n_samples, pos_dim] - Position inputs (x, y)
            map_features: [map_feature_dim] - Map features of shape [512]

        Returns:
            amp: [batch_size_new, n_samples] - Amplitude attenuation
            phase: [batch_size_new, n_samples] - Phase change
            features: [batch_size_new, n_samples, feature_dim] - Feature vectors
        """
        batch_size_new, n_samples, _ = x.size()
        x = x.view(-1, x.size(-1))  # [batch_size_new * n_samples, pos_dim]

        # map_features: [map_feature_dim]
        # Expand map_features to match the batch size and number of samples
        map_features_expanded = map_features.unsqueeze(0).expand(x.size(0), -1)  # [batch_size_new * n_samples, map_feature_dim]

        x_embed = self.embed_fn(x)  # Positional encoding
        initial_input = torch.cat([x_embed, map_features_expanded], dim=-1)

        x = F.relu(self.fc1(initial_input))

        for i, layer in enumerate(self.fc_layers):
            if i == 4:  # Skip connection at the 5th layer
                x = torch.cat([x, initial_input], dim=-1)
            x = F.relu(layer(x))

        features = self.feature_output(x).view(batch_size_new, n_samples, -1)
        amp = self.amp_output(x).view(batch_size_new, n_samples)
        phase = self.phase_output(x).view(batch_size_new, n_samples)

        return amp, phase, features

