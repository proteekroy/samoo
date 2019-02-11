from itertools import product
from torch.autograd import Variable
from utils import *
from dataloader import *
import numpy as np
from metamodels.neural_metamodel import NeuralMetamodel


class SiameseMetamodel(NeuralMetamodel):

    def __init__(self, n_var, n_obj,
                 problem_name='problem_obj',
                 n_splits=20,
                 embed_length=10,
                 batch_size=10,
                 total_epoch=200,
                 resume=False,
                 cross_val=False,
                 resume_filename=None,
                 neuralnet=None,
                 disp = True,
                 best_accuracy_model=True,
                 save_model=False,
                 dataset_func=True
                 ):
        super().__init__(n_var, n_obj, problem_name, n_splits, embed_length,
                         batch_size, total_epoch, resume, cross_val,
                         resume_filename, neuralnet, disp, best_accuracy_model,
                         save_model, dataset_func)

    def predict(self, input, *args, **kwargs):
        self.model.eval()
        input = torch.from_numpy(input)
        output, _, _ = self.model.forward(Variable(input.float()), Variable(input.float()))
        return output.data.numpy()

    def perform_epoch(self, epoch, test_flag=False):
        if test_flag:
            self.net.eval()
            loader = self.testloader
        else:
            self.net.train()
            loader = self.trainloader

        losses, top = AverageMeter(), AverageMeter()

        for batch_idx, (train_index, x, f) in enumerate(loader):

            self.batch_size = f.shape[0]
            index = torch.from_numpy(
                np.asarray(list(product(np.asarray(range(0, self.batch_size)), np.asarray(range(0, self.batch_size))))))

            f1 = f[index[:, 0]]
            f2 = f[index[:, 1]]
            x1 = x[index[:, 0]]
            x2 = x[index[:, 1]]
            label = (f1 <= f2).float()

            if self.use_cuda:
                x, x1, x2, f, f1, f2, label = Variable(x.float().cuda()), Variable(x1.cuda()), \
                                              Variable(x2.cuda()), Variable(f.float().cuda()), \
                                              Variable(f1.cuda()), Variable(f2.cuda()), Variable(label.cuda())
            else:
                x, x1, x2, f, f1, f2 = Variable(x), Variable(x1), Variable(x2), \
                                       Variable(f.float()), Variable(f1), Variable(f2)

            self.optimizer.zero_grad()
            _, _, predicted_label = self.net.forward(x1.float(), x2.float())
            predicted_f, _, _ = self.net.forward(x.float(), x.float())
            loss1 = self.mseLoss(predicted_f, f.float())
            # loss2 = self.mseLoss(predicted_label.float(), label.float())
            loss2 = self.BCEloss(predicted_label.float(), label.float())
            # out_label = (torch.flatten(out_label)).view(-1,1)
            # label = (torch.flatten(label)).view(-1,1)
            # loss3 = self.crossEntropyLoss(out_label.float(), label.long())
            loss = loss1 + loss2
            if not test_flag:
                loss.backward()
                self.optimizer.step()

            prec_matrix = torch.eq(torch.round(predicted_label.data).long(), label.long())
            prec_f = 100*torch.mean(prec_matrix.float())

            # measure accuracy and record loss
            losses.update(loss.data.item(), 1)
            top.update(prec_f, 1)
            if self.disp:
                if test_flag:
                    print('Test Epoch: %d  | Loss : %.4f | Acc : %.4f ' % (epoch, losses.avg, top.avg))
                else:
                    print('Train Epoch: %d  | Loss : %.4f | Acc : %.4f ' % (epoch, losses.avg, top.avg))

        return losses.avg, top.avg
