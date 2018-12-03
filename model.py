import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):

    def __init__(self, n_class=3, n_var=30, hidden_layer_length=10,embed_length=5):
        super(Siamese, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(n_var),
            nn.Linear(n_var, hidden_layer_length),
            nn.BatchNorm1d(hidden_layer_length),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_layer_length),
            nn.Linear(hidden_layer_length, embed_length),
            nn.BatchNorm1d(embed_length),
            nn.PReLU(),
            # nn.Dropout(),
        )
        # self.linear = nn.Linear(embed_length, n_class)
        # self.out = F.tanh(n_class)

    def forward_one(self, x):
        x = self.model(x)
        # x = x.view(x.size()[0], -1)
        # x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        out = torch.abs(out1 - out2)
        # out = self.out(dis)
        # return torch.sigmoid(out)
        return out1, out2
