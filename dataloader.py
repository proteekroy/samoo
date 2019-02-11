from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
import torch
import pickle
import torchvision
import torchvision.transforms as transforms
import math
from dataset import DatasetOpt


class DataLoader:

    def __init__(self, data_x=None, data_f=None, data_g=None, train_data_x=None,
                 test_data_x=None, train_data_f=None, test_data_f=None,
                 train_data_g=None, test_data_g=None):
        self.train_data_x = train_data_x
        self.test_data_x = test_data_x
        self.train_data_f = train_data_f
        self.test_data_f = test_data_f
        self.train_data_g = train_data_g
        self.test_data_g = test_data_g
        self.data_x = data_x
        self.data_f = data_f
        self.data_g = data_g

    def calc_domination_matrix(self, F, _F=None, epsilon=0.0):

        if _F is None:
            _F = F

        # look at the obj for dom
        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

        M = np.logical_and(smaller, np.logical_not(larger)) * 1 \
            + np.logical_and(larger, np.logical_not(smaller)) * -1
        M[M < 0] = 2

        return M

    def load(self, problem_name='DTLZ1'):
        self.data_x = np.loadtxt('Test/'+str(problem_name)+'.x')
        self.data_f = np.loadtxt('Test/'+str(problem_name)+'.f')
        self.data_g = np.loadtxt('Test/'+str(problem_name)+'.cv')

        self.data_x = (self.data_x - self.data_x.min(axis=0)) / (self.data_x.max(axis=0)+1e-16 - self.data_x.min(axis=0))
        self.data_f = (self.data_f - self.data_f.min(axis=0)) / (self.data_f.max(axis=0)+1e-16 - self.data_f.min(axis=0))
        self.data_g = (self.data_g - self.data_g.min(axis=0)) / (self.data_g.max(axis=0)+1e-16 - self.data_g.min(axis=0))

        n = math.ceil(self.data_x.shape[0] / 2)

        self.train_data_x = self.data_x[0:n, :]
        self.test_data_x = self.data_x[n:self.data_x.shape[0], :]

        self.train_data_f = self.data_f[0:n, :]
        self.test_data_f = self.data_f[n:self.data_f.shape[0], :]

        self.train_data_g = self.data_g[0:n]
        self.test_data_g = self.data_g[n:self.data_g.shape[0]]

        self.train_size = self.train_data_x.shape[0]
        self.test_size = self.test_data_x.shape[0]

        self.train_dominance_matrix = self.calc_domination_matrix(self.train_data_f)
        self.test_dominance_matrix = self.calc_domination_matrix(self.test_data_f)

        # self.train_comparison_matrix = np.zeros((self.train_size, self.data_f.shape[1]))
        # self.test_comparison_matrix = np.zeros((self.test_size, self.data_f.shape[1]))
        #
        # for i in range(0, self.data_f.shape[1]):
        #     self.train_comparison_matrix[:, i] = self.calc_domination_matrix(self.train_data_f[:, [i]])
        #     self.test_comparison_matrix[:, i] = self.calc_domination_matrix(self.test_data_f[:, [i]])

        self.train_data_x = torch.from_numpy(self.train_data_x).float()
        self.test_data_x = torch.from_numpy(self.test_data_x).float()

        self.train_data_f = torch.from_numpy(self.train_data_f).float()
        self.test_data_f = torch.from_numpy(self.test_data_f).float()

        self.train_data_g = torch.from_numpy(self.train_data_g).float()
        self.test_data_g = torch.from_numpy(self.test_data_g).float()

        train_indices = torch.from_numpy(np.asarray(range(0, self.train_size)))
        test_indices = torch.from_numpy(np.asarray(range(0, self.test_size)))

        self.trainset = DatasetOpt(train_indices, self.train_data_x, self.train_data_f, self.train_data_g)
        self.testset = DatasetOpt(test_indices, self.test_data_x, self.test_data_f, self.test_data_g)

        self.train_label_zero = (self.train_dominance_matrix == 0).sum()/(self.train_size*self.train_size)
        self.train_label_one = (self.train_dominance_matrix == 1).sum()/(self.train_size*self.train_size)
        self.train_label_two = (self.train_dominance_matrix == 2).sum()/(self.train_size*self.train_size)

        self.test_label_zero = (self.test_dominance_matrix == 0).sum() / (self.test_size * self.test_size)
        self.test_label_one = (self.test_dominance_matrix == 1).sum() / (self.test_size * self.test_size)
        self.test_label_two = (self.test_dominance_matrix == 2).sum() / (self.test_size * self.test_size)

        self.embed_length = self.data_f.shape[1]
