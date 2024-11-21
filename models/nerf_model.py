import torch.nn as nn

from .attenuation_network import AttenuationNetwork
from .radiance_network import RadianceNetwork

class NeRFModel(nn.Module):
    def __init__(self, map_features, map_feature_dim=512, pos_dim=2, direction_dim=2):
        super(NeRFModel, self).__init__()
        self.map_features = map_features.clone().detach().requires_grad_(False)

        self.attenuation_net = AttenuationNetwork(pos_dim, map_feature_dim)
        self.radiance_net = RadianceNetwork(pos_dim, direction_dim)

    def forward(self, position, direction):
        amp, phase, attenuation_features = self.attenuation_net(position, self.map_features)
        radiance = self.radiance_net(position, direction, attenuation_features)
        return amp, phase, radiance
