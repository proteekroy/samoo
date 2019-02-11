from metamodels.neural_metamodel import NeuralMetamodel
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import copy
from torch.autograd import Variable
from utils import *
from dataset import *
import torch
import numpy as np
from frameworks.normalize import NormalizeConstraint


class ConstraintMetamodel(NeuralMetamodel):

    def __init__(self, n_var, n_obj,
                 problem_name='problem_constr',
                 n_splits=10,
                 embed_length=10,
                 batch_size=10,
                 total_epoch=200,
                 resume=False,
                 cross_val=False,
                 resume_filename=None,
                 neuralnet=None,
                 disp = False,
                 best_accuracy_model=True,
                 save_model=False,
                 dataset_func=False
                 ):
        super().__init__(n_var, n_obj, problem_name, n_splits, embed_length,
                         batch_size, total_epoch, resume, cross_val,
                         resume_filename, neuralnet, disp, best_accuracy_model,
                         save_model, dataset_func)

    def predict(self, input, *args, **kwargs):
        self.model.eval()
        input = torch.from_numpy(input)
        _, z = self.model(Variable(input.float()))
        return z.data.numpy()

    def perform_epoch(self, epoch, test_flag=False):
        if test_flag:
            self.net.eval()
            loader = self.testloader
        else:
            self.net.train()
            loader = self.trainloader

        losses, top = AverageMeter(), AverageMeter()

        for batch_idx, (train_index, x, g, label) in enumerate(loader):

            label = label.float()
            if self.use_cuda:
                x, g, label = Variable(x.float().cuda()),Variable(g.float().cuda()), Variable(label.cuda())
            else:
                x, g, label = Variable(x), Variable(g.float()), Variable(label.long())

            # learn regression+classification
            self.optimizer.zero_grad()
            predicted_label, z = self.net(x.float())
            loss1 = self.mseLoss(z, g)
            loss2 = self.mseLoss(predicted_label.float(), label.float())
            # loss2 = self.BCEloss(predicted_label.float(), label.float())
            loss = loss1 + loss2
            if not test_flag:
                loss.backward()
                self.optimizer.step()

            # measure accuracy and record loss
            prec_matrix = torch.eq(torch.round(predicted_label.data).long(), label.long())
            prec_f = 100 * torch.mean(prec_matrix.float())

            # measure accuracy and record loss
            losses.update(loss.data.item(), 1)
            top.update(prec_f, 1)

            if self.disp:
                if test_flag:
                    print('Test Epoch: %d  | Loss : %.4f | Acc : %.4f ' % (epoch, losses.avg, top.avg))
                else:
                    print('Train Epoch: %d  | Loss : %.4f | Acc : %.4f ' % (epoch, losses.avg, top.avg))

        return losses.avg, top.avg


