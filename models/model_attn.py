import torch
from torch import nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.u = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.w = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.w(self.v(x) * self.u(x))
        s = torch.softmax(w, dim=0).T
        x = torch.mm(s, x)
        return x


class Model(nn.Module):
    def __init__(self, input_dim=768, embed_dim=64, num_classes=4):
        super(Model, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.attention = AttentionLayer(embed_dim, embed_dim * 4, 1)
        self.predictor = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        a = self.embedding(x.squeeze(0))
        a = self.attention(a)
        out = self.predictor(a)
        return out
