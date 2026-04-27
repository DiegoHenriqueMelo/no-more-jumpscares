import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class MultimodalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_img = torch.zeros(1, 1, 84, 84)
            cnn_output_dim = self.cnn(sample_img).shape[1]

        self.fc_estados = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(cnn_output_dim + 32, 256),
            nn.ReLU(),
        )

    def forward(self, observations):
        imagem = observations["imagem"].float() / 255.0
        estados = observations["estados"]

        if imagem.dim() == 3:
            imagem = imagem.unsqueeze(0)

        if imagem.shape[-1] == 1:
            imagem = imagem.permute(0, 3, 1, 2)
        elif imagem.shape[1] != 1:
            imagem = imagem.permute(0, 3, 1, 2)

        features_img = self.cnn(imagem)
        features_estados = self.fc_estados(estados)

        combined = torch.cat([features_img, features_estados], dim=1)
        return self.fc_combined(combined)
