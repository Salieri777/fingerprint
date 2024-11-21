import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.positional_encoding import get_embedder


class RadianceNetwork(nn.Module):
    def __init__(self, pos_dim, dir_dim, feature_dim = 256, hidden_dim1=256, hidden_dim2=128, output_dim=1, multires_pos=10, multires_dir=10):
        super(RadianceNetwork, self).__init__()

        self.embed_pos_fn, pos_embed_dim = get_embedder(multires_pos, pos_dim)
        self.embed_dir_fn, dir_embed_dim = get_embedder(multires_dir, dir_dim)

        input_dim = pos_embed_dim + dir_embed_dim + feature_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, position, direction, features):
        """
        :param position: [batch_size_new, n_samples, pos_dim]
        :param direction: [batch_size_new, n_samples, dir_dim]
        :param features: [batch_size_new, n_samples, feature_dim]
        :return: 信号角度 (angle) 和振幅 (abs)
        """
        batch_size_new, n_samples, _ = position.size()
        position = position.view(-1, position.size(-1))    # [batch_size_new * n_samples, pos_dim]
        direction = direction.view(-1, direction.size(-1))  # [batch_size_new * n_samples, dir_dim]
        features = features.view(-1, features.size(-1))    # [batch_size_new * n_samples, feature_dim]

        position_encoded = self.embed_pos_fn(position)
        direction_encoded = self.embed_dir_fn(direction)

        x = torch.cat([position_encoded, direction_encoded, features], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.output_layer(x).view(batch_size_new, n_samples, -1)

        return output
