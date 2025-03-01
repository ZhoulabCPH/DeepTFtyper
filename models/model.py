import torch
from torch import nn
from torch_geometric.nn.dense import DMoNPooling
from torch_geometric.nn import SAGEConv, GraphNorm
from torch_geometric.utils import remove_isolated_nodes, to_dense_batch, to_dense_adj

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class ConvLayer(nn.Module):
    def __init__(self, dim, num_nodes):
        super(ConvLayer, self).__init__()
        self.conv1 = SAGEConv(dim, dim)
        self.conv2 = SAGEConv(dim, dim)

        self.co_conv = SAGEConv(dim, dim)
        self.co_norm = GraphNorm(dim)

        self.conv3 = SAGEConv(dim, dim)
        self.conv4 = SAGEConv(dim, dim)

        self.activate = nn.Mish()
        self.pooling = DMoNPooling([dim, dim], num_nodes)

    def forward(self, x1, x2, e1, e2):
        x1 = self.conv1(x1, e1)
        x2 = self.conv2(x2, e2)
        x1 = self.activate(x1)
        x2 = self.activate(x2)

        c1 = self.co_conv(x1, e1)
        c2 = self.co_conv(x2, e2)
        c1 = self.activate(c1)
        c2 = self.activate(c2)

        x1 = self.conv3(x1 + c1, e1)
        x2 = self.conv4(x2 + c2, e2)
        x1 = self.activate(x1)
        x2 = self.activate(x2)

        x1, mask = to_dense_batch(x1)
        adj = to_dense_adj(e1)
        _, x1, _, sp1, o1, c1 = self.pooling(x1, adj, mask)

        x2, mask = to_dense_batch(x2)
        adj = to_dense_adj(e2)
        _, x2, _, sp2, o2, c2 = self.pooling(x2, adj, mask)

        x = torch.cat((x1, x2), dim=1)

        return x, sp1 + o1 + c1 + sp2 + o2 + c2


class SwiGLULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(SwiGLULayer, self).__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, output_dim)
        self.activate = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.dropout(self.activate(self.w1(x))) * self.w2(x))


class Model(nn.Module):
    def __init__(self, input_dim=768, embed_dim=64, num_classes=4, dropout=0.3):
        super(Model, self).__init__()
        self.d_loss = 0.
        self.embedding_layer = SwiGLULayer(input_dim, input_dim // 2, embed_dim, dropout)
        self.conv_layer = ConvLayer(embed_dim, 300)
        self.predictor = SwiGLULayer(embed_dim, embed_dim // 2, num_classes, dropout)

    def forward(self, x, e1, e2):
        x = self.embedding_layer(x.squeeze(0))
        e1, _, mask1 = remove_isolated_nodes(edge_index=e1, num_nodes=len(x))
        e2, _, mask2 = remove_isolated_nodes(edge_index=e2, num_nodes=len(x))
        x, self.d_loss = self.conv_layer(x[mask1], x[mask2], e1, e2)
        output = self.predictor(torch.mean(x, dim=1))

        return output
