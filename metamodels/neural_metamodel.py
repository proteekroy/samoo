import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from pymoo.util.dominator import Dominator
import datetime
from dataset import *
from abc import abstractmethod
import copy
from torch.autograd import Variable
import os
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from utils import *
import numpy as np
# from metamodels.regression_metamodel import *
from frameworks.normalize import NormalizeConstraint


class MetaModel:
    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def train(self, input, target, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, input, *args, **kwargs):
        pass


class NeuralMetamodel(MetaModel):

    def __init__(self, n_var, n_obj,
                 problem_name='Problem',
                 n_splits=10,
                 embed_length=10,
                 batch_size=10,
                 total_epoch=200,
                 resume=False,
                 cross_val=False,
                 resume_filename=None,
                 neuralnet=None,
                 disp=False,
                 best_accuracy_model=True,
                 save_model=False,
                 dataset_func=True
                 ):
        if resume:
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(resume_filename)
            self.neuralnet = checkpoint['net']
            self.best_acc = checkpoint['acc']
            self.best_loss = checkpoint['loss']
            self.start_epoch = checkpoint['epoch']
            self.total_epoch = total_epoch + self.start_epoch
        else:
            self.neuralnet = neuralnet # Siamese(n_var=n_var, n_obj=n_obj, embed_length=embed_length)
            self.best_acc = 0  # best test accuracy_f
            self.best_loss = np.inf
            self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
            self.total_epoch = total_epoch

        self.n_var = n_var
        self.n_obj = n_obj
        self.n_splits = n_splits
        self.problem_name = problem_name
        self.embed_length = embed_length
        self.batch_size = batch_size  # 16
        self.cross_val = cross_val
        self.disp = disp
        self.best_accuracy_model = best_accuracy_model
        self.save_model = save_model
        self.dataset_func = dataset_func
        self.checkpoint_filename = str(self.problem_name).lower()
        self.log_file_name = self.checkpoint_filename + '_log.txt'
        self.use_cuda = torch.cuda.is_available()
        self.normalize = NormalizeConstraint()

        if self.use_cuda:
            self.net.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        self.crossEntropyLoss = nn.CrossEntropyLoss()
        self.mseLoss = nn.MSELoss()
        self.BCEloss = nn.BCELoss()

        if self.save_model:
            self.logger = Logger(os.path.join('../checkpoint/', self.log_file_name), title=self.problem_name,
                                 resume=resume)
            self.logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'Train Acc.', 'Test Acc.'])

        super().__init__()

    def train(self, x, f, cross_val=False, *args, **kwargs):

        # f = (f - np.min(f))/(np.max(f)-np.min(f))
        f_normalized = self.normalize.normalize(f, self.dataset_func)
        kf = KFold(n_splits=self.n_splits)
        best_acc = 0
        best_loss = np.inf

        if self.dataset_func is False:
            cv = np.copy(f_normalized)
            index = np.any(f_normalized > 0, axis=1)
            cv[f_normalized <= 0] = 0
            cv = np.sum(cv, axis=1)
            acv = np.sum(f_normalized, axis=1)
            acv[index] = np.copy(cv[index])
            g_label = cv > 0
            g_label[cv <= 0] = -1
            # g_label = g > 0
            # g_label[g <= 0] = -1
            g_label = g_label.astype(int)
            g_label = np.vstack(g_label)

        # cross-validation
        for train_index, test_index in kf.split(x):
            train_data_x, test_data_x, train_data_f, test_data_f \
                = x[train_index], x[test_index], f_normalized[train_index], f_normalized[test_index]

            if self.dataset_func:
                self.train_dominance_matrix = Dominator.calc_domination_matrix(f_normalized[train_index])
                self.test_dominance_matrix = Dominator.calc_domination_matrix(f_normalized[test_index])
            else:
                train_data_cv, test_data_cv, train_g_label, test_g_label, \
                = torch.from_numpy(cv[train_index]), torch.from_numpy(cv[test_index]), \
                  torch.from_numpy(g_label[train_index]), torch.from_numpy(g_label[test_index])

            train_data_x, test_data_x, train_data_f, test_data_f \
                = torch.from_numpy(train_data_x), torch.from_numpy(test_data_x), torch.from_numpy(train_data_f), \
                  torch.from_numpy(test_data_f)

            train_indices = torch.from_numpy(np.asarray(range(0, train_data_x.shape[0])))
            test_indices = torch.from_numpy(np.asarray(range(0, test_data_x.shape[0])))

            if self.dataset_func:
                self.trainset = DatasetFunction(train_indices, train_data_x, train_data_f)
                self.testset = DatasetFunction(test_indices, test_data_x, test_data_f)
            else:
                self.trainset = DatasetConstraint(train_indices, train_data_x, train_data_f, train_g_label)
                self.testset = DatasetConstraint(test_indices, test_data_x, test_data_f, test_g_label)

            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)

            self.net = copy.deepcopy(self.neuralnet)
            self.optimizer = optim.Adam(self.net.parameters(), lr=0.01, weight_decay=5e-1, betas=(0.9, 0.999))
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.total_epoch, eta_min=1e-7)

            model, acc, loss = self.train_partition()
            if self.best_accuracy_model:
                if acc > best_acc:
                    self.model = model
            else:
                if loss < best_loss:
                    self.model = model

            if not self.cross_val:
                break

        return self.model

    def train_partition(self):  # Train one partition of cross-validation
        self.best_net = self.net
        for epoch in range(self.start_epoch, self.start_epoch + self.total_epoch):
            if self.disp:
                print('\nEpoch: %d' % epoch)
            self.scheduler.step()
            train_loss, train_acc = self.perform_epoch(epoch, test_flag=False)
            test_loss, test_acc = self.perform_epoch(epoch, test_flag=True)
            if self.save_model:
                self.logger.append([self.optimizer.param_groups[0]['lr'], float(train_loss), float(test_loss),
                                    float(train_acc), float(test_acc)])
            # Keep Best Model.
            if self.best_accuracy_model:
                if test_acc > self.best_acc:
                    self.best_net = self.net.module if self.use_cuda else self.net
                    self.best_acc = test_acc
                    self.best_loss = test_loss
                    if self.save_model:
                        if self.disp:
                            print('Saving..')
                        state = {
                            'net': self.net.module if self.use_cuda else self.net,
                            'state_dict': self.net.state_dict(),
                            'acc': test_acc,
                            'epoch': epoch,
                            'optimizer': self.optimizer.state_dict()
                        }
                        if not os.path.isdir('checkpoint'):
                            os.mkdir('checkpoint')
                        torch.save(state, 'checkpoint/' + self.checkpoint_filename+'.ckpt')
            else:
                if test_loss < self.best_loss:
                    self.best_net = self.net.module if self.use_cuda else self.net
                    self.best_acc = test_acc
                    self.best_loss = test_loss

                    if self.save_model:
                        if self.disp:
                            print('Saving..')
                        state = {
                            'net': self.net.module if self.use_cuda else self.net,
                            'state_dict': self.net.state_dict(),
                            'acc': test_acc,
                            'epoch': epoch,
                            'optimizer': self.optimizer.state_dict()
                        }
                        if not os.path.isdir('checkpoint'):
                            os.mkdir('checkpoint')
                        torch.save(state, 'checkpoint/' + self.checkpoint_filename+'.ckpt')
        if self.save_model:
            self.logger.close()

        return self.best_net, self.best_acc, self.best_loss

    @abstractmethod
    def predict(self, input, *args, **kwargs):
        pass

    @abstractmethod
    def perform_epoch(self, epoch, test_flag=False):
        pass
