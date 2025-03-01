import torch
from torch import nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class Model(nn.Module):
    def __init__(self, input_dim=768, embed_dim=64, num_classes=4):
        super(Model, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, embed_dim)
        )
        self.predictor = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.mean(x, dim=1)
        out = self.predictor(x)
        return out
