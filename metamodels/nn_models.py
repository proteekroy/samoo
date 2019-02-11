import torch.nn as nn
import copy
from dataloader import *


class TruncatedGaussianActivation(nn.Module):

    def __init__(self, mean=0, std=1, min=0.1, max=0.9):
        super(TruncatedGaussianActivation, self).__init__()
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max

    def forward(self, x):
        gauss = torch.exp((-(x - self.mean) ** 2)/(2* self.std ** 2))
        return torch.clamp(gauss, min=self.min, max=self.max)


class QuadraticActivation(nn.Module):

    def forward(self, x):
        return torch.pow(x, 2)


class CubicActivation(nn.Module):

    def forward(self, x):
        return torch.pow(x, 3)


class SineActivation(nn.Module):

    def forward(self, x):
        return torch.sin(x)


class CoSineActivation(nn.Module):

    def forward(self, x):
        return torch.cos(x)


class TanActivation(nn.Module):

    def forward(self, x):
        return torch.tan(x)


class MixedActivation(nn.Module):

    def __init__(self, length):
        super(MixedActivation, self).__init__()
        self.length = length
        self.activation_list = dict()
        self.activation_list['1'] = QuadraticActivation()  #SineActivation()
        self.activation_list['2'] = QuadraticActivation()  #
        self.activation_list['3'] = QuadraticActivation()  #TruncatedGaussianActivation()  # CubicActivation()
        self.activation_list['4'] = nn.PReLU()  # CubicActivation()
        self.activation_list['5'] = nn.PReLU()  # CoSineActivation()
        self.activation_list['6'] = nn.PReLU()  # TanActivation()
        self.activation_profile = dict()

        for i in range(0, self.length):
            t = i % len(self.activation_list)
            self.activation_profile[str(i+1)] = self.activation_list[str(t+1)]

    def forward(self, x):
        temp = dict()
        for i in range(0, self.length):
            temp[str(i+1)] = self.activation_profile[str(i+1)](x[:, i])

        out = temp['1'].view(-1, 1)
        for i in range(1, self.length):
            out = torch.cat((out, temp[str(i + 1)].view(-1, 1)), 1)

        return out


class SiameseNet(nn.Module):

    def __init__(self, n_var=6, n_obj=2, hidden_layer_length=15, embed_length=5):
        self.n_var = n_var
        self.n_obj = n_obj
        self.embed_length = embed_length
        self.hidden_layer_length = hidden_layer_length
        super(SiameseNet, self).__init__()

        self.shared = nn.Sequential(
            nn.BatchNorm1d(n_var),
            nn.Linear(n_var, hidden_layer_length),
            # nn.BatchNorm1d(hidden_layer_length),
            # nn.PReLU(),
            # TruncatedGaussianActivation(),
            QuadraticActivation(),
            # MixedActivation(self.hidden_layer_length),
            nn.Dropout(),
            # nn.PReLU(),
            nn.Linear(self.hidden_layer_length, self.embed_length),
            # nn.BatchNorm1d(self.embed_length),
            # MixedActivation(self.embed_length),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(self.embed_length, self.embed_length),
            # nn.BatchNorm1d(self.embed_length),
            # nn.ELU()
            # TruncatedGaussianActivation(),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(self.embed_length, self.n_obj)
            # nn.Dropout(),
        )
        self.merge = nn.Sequential(
            nn.BatchNorm1d(2*self.n_obj),
            nn.Linear(2*self.n_obj, self.n_obj),
            nn.Sigmoid(),
        )

        self.func = dict()
        for i in range(0, self.n_obj):
            self.func[str(i+1)] = self.create_objective_layer()

    def forward_one(self, x):
        z = self.shared(x)
        # x = x.view(x.size()[0], -1)
        # x = self.linear(x)
        return z

    def forward(self, x1, x2):
        '''
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)

        f1_dict = dict()
        f2_dict = dict()
        for i in range(0, self.n_obj):
            f1_dict[str(i + 1)] = self.func[str(i + 1)](out1)
            f2_dict[str(i + 1)] = self.func[str(i + 1)](out2)
        #
        f1 = f1_dict['1']
        f2 = f2_dict['1']
        for i in range(1, self.n_obj):
            f1 = torch.cat((f1, f1_dict[str(i + 1)]), 1)
            f2 = torch.cat((f2, f2_dict[str(i + 1)]), 1)

        f_cat = torch.cat((f1, f2), 1)
        out_label = self.merge(f_cat)
        '''
        f1 = self.shared(x1)
        f2 = self.shared(x2)

        f_cat = torch.cat((f1, f2), 1)
        out_label = self.merge(f_cat)

        return f1, f2, out_label

    def create_objective_layer(self):
        func = nn.Sequential(
            # nn.BatchNorm1d(self.hidden_layer_length2),
            # nn.Linear(self.hidden_layer_length2, self.embed_length),
            # nn.BatchNorm1d(self.embed_length),
            # nn.PReLU(),
            nn.Linear(self.embed_length, 1),
            # nn.BatchNorm1d(1),
            # nn.PReLU(),
            # TruncatedGaussianActivation()
        )
        return func


class ConstraintNet(nn.Module):

    def __init__(self, n_var=6, n_constr=6, hidden_layer_length=3, embed_length=5):

        self.n_var = n_var
        self.n_constr = n_constr
        self.embed_length = embed_length
        self.hidden_layer_length = hidden_layer_length
        super(ConstraintNet, self).__init__()

        self.shared = nn.Sequential(
            nn.BatchNorm1d(n_var),
            nn.Linear(n_var, hidden_layer_length),
            nn.BatchNorm1d(hidden_layer_length),
            # nn.PReLU(),
            QuadraticActivation(),
            nn.Linear(hidden_layer_length, hidden_layer_length),
            nn.BatchNorm1d(hidden_layer_length),
            nn.PReLU(),
            # nn.Dropout(),
        )
        self.func = dict()
        for i in range(0, self.n_constr):
            self.func[str(i + 1)] = self.create_constraint_layer()

        self.merge = nn.Sequential(
            nn.BatchNorm1d(self.n_constr),
            nn.Linear(self.n_constr, 1),
            nn.Sigmoid(),
        )
        # self.classifier = nn.Sequential(nn.Linear(n_constr, 2))
        # self.output = nn.Tanh()

    def forward(self, x):
        z = self.shared(x)

        g = dict()
        for i in range(0, self.n_constr):
            g[str(i + 1)] = self.func[str(i + 1)](z)

        z = torch.cat([g[str(i + 1)] for i in range(0, self.n_constr)], 1)

        out = self.merge(z)

        return out, z

    def create_constraint_layer(self):
        func = nn.Sequential(
            # nn.BatchNorm1d(self.hidden_layer_length),
            # nn.Linear(self.hidden_layer_length, self.hidden_layer_length),
            # nn.BatchNorm1d(self.hidden_layer_length),
            # nn.PReLU(),
            nn.Linear(self.hidden_layer_length, 1),
            nn.PReLU()
        )
        return copy.deepcopy(func)


class ConstraintNet2(nn.Module):

    def __init__(self, n_var=6, n_constr=6, hidden_layer_length=3, embed_length=5):

        self.n_var = n_var
        self.n_constr = n_constr
        self.embed_length = embed_length
        self.hidden_layer_length = hidden_layer_length
        super(ConstraintNet2, self).__init__()

        self.shared = nn.Sequential(
            nn.BatchNorm1d(n_var),
            nn.Linear(n_var, hidden_layer_length),
            nn.BatchNorm1d(hidden_layer_length),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_layer_length),
            nn.Linear(hidden_layer_length, hidden_layer_length),
            nn.BatchNorm1d(hidden_layer_length),
        )
        self.regressor = nn.Sequential(
            nn.PReLU(),
            nn.BatchNorm1d(hidden_layer_length),
            nn.Linear(hidden_layer_length, 1))
        self.classifier = nn.Sequential(nn.Linear(hidden_layer_length, 2))

    def forward(self, x):
        z = self.shared(x)
        out1 = self.regressor(z)
        out2 = self.classifier(z)

        return out2, out1.flatten()

    def create_constraint_layer(self):
        func = nn.Sequential(
            nn.BatchNorm1d(self.hidden_layer_length),
            nn.Linear(self.hidden_layer_length, self.hidden_layer_length),
            nn.BatchNorm1d(self.hidden_layer_length),
            nn.PReLU(),
            nn.Linear(self.hidden_layer_length, 1),
        )
        return func


class RepresentationNet(nn.Module):

    def __init__(self, n_class=2, n_var=30, hidden_size=10, embed_length=5):
        super(RepresentationNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(n_var, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(embed_length),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(embed_length, embed_length),
            nn.ReLU(),
            nn.Linear(embed_length, embed_length),
            nn.ReLU(),
            nn.Linear(embed_length, embed_length),
            nn.ReLU(),
        )

        self.shortcut = nn.Sequential(nn.Linear(embed_length, embed_length))
        self.classlayer = nn.Linear(embed_length, n_class)

    def forward(self, x):
        z = self.block1(x)
        z = z + self.shortcut(x)
        z = self.block2(z)
        out = self.classlayer(z)

        return out, z


class SelectionNet(nn.Module):

    def __init__(self, n_obj=3, n_var=30, hidden_layer_length=10,embed_length=5):
        super(SelectionNet, self).__init__()

        self.model_base = nn.Sequential(
            nn.BatchNorm1d(n_var),
            nn.Linear(n_var, hidden_layer_length),
            nn.PReLU(),
            nn.Linear(hidden_layer_length, embed_length),
            nn.BatchNorm1d(embed_length),
            nn.PReLU(),
        )

        self.model_framework12 = nn.Sequential(
            nn.BatchNorm1d(embed_length),
            nn.Linear(embed_length, 5),
            nn.PReLU(),
            nn.Linear(5, n_obj),
        )

        self.model_framework22 = nn.Sequential(
            nn.BatchNorm1d(embed_length),
            nn.Linear(embed_length, 5),
            nn.PReLU(),
            nn.Linear(5, n_obj),
        )

        self.model_framework32 = nn.Sequential(
            nn.BatchNorm1d(embed_length),
            nn.Linear(embed_length, 5),
            nn.PReLU(),
            nn.Linear(5, n_obj),
        )

        self.model_framework42 = nn.Sequential(
            nn.BatchNorm1d(embed_length),
            nn.Linear(embed_length, 5),
            nn.PReLU(),
            nn.Linear(5, n_obj),
        )

        self.model_framework6 = nn.Sequential(
            nn.BatchNorm1d(embed_length),
            nn.Linear(embed_length, 5),
            nn.PReLU(),
            nn.Linear(5, n_obj),
        )

    def forward_base(self, x):
        x = self.model(x)
        # x = x.view(x.size()[0], -1)
        # x = self.linear(x)
        return x

    def forward(self, x):
        z = self.forward_base(x)
        out1 = self.model_framework12(z)
        out2 = self.model_framework22(z)
        out3 = self.model_framework32(z)
        out4 = self.model_framework42(z)
        out5 = self.model_framework6(z)

        return out1, out2, out3, out4, out5
