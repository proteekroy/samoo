import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import Dataset
from utils import *
import os
import scipy.io as sio
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import Dataset
from utils import *
import os
# from network import *
import pandas as pd
import numpy as np
import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import torch.nn as nn
import scipy.io as sio
from metamodels import *
from dataloader import *
from itertools import product


problem_name = 'ZDT4_10_2'
n_var = 10
batch_size = 10  # 16
use_cuda = False  # torch.cuda.is_available()
#embed_length = 64
total_epoch = 200
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
gamma_param = 0

checkpoint_filename = str(problem_name)+'.ckpt'
log_file_name = str(problem_name)+'_log.txt'


print('==> Loading and Preparing data..')
data = DataLoader()
data.load(problem_name=problem_name)

trainloader = torch.utils.data.DataLoader(data.trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(data.testset, batch_size=batch_size, shuffle=True)


print('==> Building metamodels..')
net = Siamese(n_var=n_var, embed_length=data.embed_length)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

ce_criterion = nn.CrossEntropyLoss()
mse_criterion = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3, nesterov=False)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-1, betas=(0.9, 0.999))
scheduler1 = CosineAnnealingLR(optimizer,  T_max=total_epoch, eta_min=1e-7)
if start_epoch == 0:
    logger = Logger(os.path.join('checkpoint/', log_file_name), title=problem_name)
else:
    logger = Logger(os.path.join('checkpoint/', log_file_name), title=problem_name, resume=True)

logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

# Training
def train():
    net.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    dataiter = iter(trainloader)

    for batch_idx, (index1, x, f, g) in enumerate(trainloader):

        index = torch.from_numpy(np.asarray(list(product(np.asarray(range(0, batch_size)), np.asarray(range(0, batch_size))))))
        f1 = f[index[:, 0]]
        f2 = f[index[:, 1]]
        x1 = x[index[:, 0]]
        x2 = x[index[:, 1]]
        g1 = g[index[:, 0]]
        g2 = g[index[:, 1]]

        # index2, x2, f2, g2 = dataiter.next()
        # indices = np.vstack((index1.data.numpy(),index2.data.numpy())).transpose()
        label = data.train_dominance_matrix[index[:, 0], index[:, 1]]
        # M_gt = data.calc_domination_matrix(F=f1.numpy(), _F=f2.numpy())
        # M_gt = calc_domination_matrix(F=f1,_F=f2)

        if use_cuda:
            x1, x2, f1, f2, g1, g2 = Variable(x1.cuda()), Variable(x2.cuda()), Variable(f1.cuda()), Variable(f2.cuda()), Variable(g1.cuda()), Variable(g2.cuda())
            label = Variable(label.cuda())
            x = Variable(torch.from_numpy(x).float().cuda(), requires_grad=True)
        else:
            x1, x2, f1, f2, g1, g2 = Variable(x1), Variable(x2), Variable(f1), Variable(f2), Variable(g1), Variable(g2)
            label = Variable(torch.from_numpy(label))
            x = Variable(x, requires_grad=True)

        optimizer.zero_grad()
        out_mse = net.forward_one(x)
        output = net.forward(x1, x2)
        # M_predict = data.calc_domination_matrix(F=out1_sep.data.numpy(), _F=out2_sep.data.numpy())
        # prec1 = (M_gt == M_predict).sum() / (M_predict.shape[0] * M_gt.shape[0])
        loss_sep = ce_criterion(output, label)
        # M_gt = Variable((torch.from_numpy(M_gt)).float(), requires_grad=True)
        # M_predict = Variable((torch.from_numpy(M_predict)).float(), requires_grad=True)
        # loss_sep = mse_criterion(M_gt, M_predict)
        f = (f.numpy() - f.numpy().min(axis=0)) / (f.numpy().max(axis=0)+1e-16 - f.numpy().min(axis=0))
        f = Variable(torch.from_numpy(f), requires_grad=False)
        loss_mse = mse_criterion(out_mse, f)
        # loss = criterion((torch.from_numpy(M_gt)).float(), (torch.from_numpy(M_predict)).float())
        # loss_sep = ce_criterion(M_gt, M_predict)
        loss = loss_mse + loss_sep
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss for learner
        prec1 = accuracy(output.data, label.data)
        losses.update(loss.data.item(), batch_size)
        top1.update(prec1[0], batch_size)
        # top1.update(prec1, batch_size)

        # print('Train Epoch: %d  | Loss: %.4f | Acc: %.4f ' % (epoch, losses.avg, top1.avg.cpu().numpy()))
        print('Train Epoch: %d  | Loss: %.4f | Acc: %.4f ' % (epoch, losses.avg, top1.avg))

    # return losses.avg, top1.avg.cpu().numpy()
    return losses.avg, top1.avg


def test():
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_idx, (index1, x, f, g) in enumerate(trainloader):

        index = torch.from_numpy(np.asarray(list(product(np.asarray(range(0, batch_size)), np.asarray(range(0, batch_size))))))
        f1 = f[index[:, 0]]
        f2 = f[index[:, 1]]
        x1 = x[index[:, 0]]
        x2 = x[index[:, 1]]
        g1 = g[index[:, 0]]
        g2 = g[index[:, 1]]

        # index2, x2, f2, g2 = dataiter.next()
        # indices = np.vstack((index1.data.numpy(),index2.data.numpy())).transpose()
        label = data.train_dominance_matrix[index[:, 0], index[:, 1]]
        # M_gt = data.calc_domination_matrix(F=f1.numpy(), _F=f2.numpy())
        # M_gt = calc_domination_matrix(F=f1,_F=f2)

        if use_cuda:
            x1, x2, f1, f2, g1, g2 = Variable(x1.cuda()), Variable(x2.cuda()), Variable(f1.cuda()), Variable(f2.cuda()), Variable(g1.cuda()), Variable(g2.cuda())
            label = Variable(label.cuda())
            x = Variable(torch.from_numpy(x).float().cuda(), requires_grad=True)
        else:
            x1, x2, f1, f2, g1, g2 = Variable(x1), Variable(x2), Variable(f1), Variable(f2), Variable(g1), Variable(g2)
            label = Variable(torch.from_numpy(label))
            x = Variable(x, requires_grad=True)

        optimizer.zero_grad()
        out_mse = net.forward_one(x)
        output = net.forward(x1, x2)
        # M_predict = data.calc_domination_matrix(F=out1_sep.data.numpy(), _F=out2_sep.data.numpy())
        # prec1 = (M_gt == M_predict).sum() / (M_predict.shape[0] * M_gt.shape[0])
        loss_sep = ce_criterion(output, label)
        # M_gt = Variable((torch.from_numpy(M_gt)).float(), requires_grad=True)
        # M_predict = Variable((torch.from_numpy(M_predict)).float(), requires_grad=True)
        # loss_sep = mse_criterion(M_gt, M_predict)
        f = (f.numpy() - f.numpy().min(axis=0)) / (f.numpy().max(axis=0)+1e-16-f.numpy().min(axis=0))
        f = Variable(torch.from_numpy(f), requires_grad=False)
        loss_mse = mse_criterion(out_mse, f)
        # loss = criterion((torch.from_numpy(M_gt)).float(), (torch.from_numpy(M_predict)).float())
        # loss_sep = ce_criterion(M_gt, M_predict)
        loss = loss_mse + loss_sep

        # measure accuracy and record loss for learner
        prec1 = accuracy(output.data, label.data)
        losses.update(loss.data.item(), batch_size)
        top1.update(prec1[0], batch_size)
        # top1.update(prec1, batch_size)

        # print('Train Epoch: %d  | Loss: %.4f | Acc: %.4f ' % (epoch, losses.avg, top1.avg.cpu().numpy()))
        print('Test Epoch: %d  | Loss: %.4f | Acc: %.4f ' % (epoch, losses.avg, top1.avg))

    # return losses.avg, top1.avg.cpu().numpy()
    return losses.avg, top1.avg
# Test
# def test():
#     net.eval()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#
#     for batch_idx, (index1, x, f, g) in enumerate(testloader):
#
#         index = torch.from_numpy(np.asarray(list(product(np.asarray(range(0, batch_size)), np.asarray(range(0, batch_size))))))
#         f1 = f[index[:, 0]]
#         f2 = f[index[:, 1]]
#         x1 = x[index[:, 0]]
#         x2 = x[index[:, 1]]
#         g1 = g[index[:, 0]]
#         g2 = g[index[:, 1]]
#
#         M_gt = data.calc_domination_matrix(F=f1.numpy(), _F=f2.numpy())
#
#         if use_cuda:
#             x1, x2, f1, f2, g1, g2 = Variable(x1.cuda()), Variable(x2.cuda()), Variable(f1.cuda()), Variable(f2.cuda()), Variable(g1.cuda()), Variable(g2.cuda())
#             # label = Variable(label.cuda())
#         else:
#             x1, x2, f1, f2, g1, g2 = Variable(x1), Variable(x2), Variable(f1), Variable(f2), Variable(g1), Variable(g2)
#             # label = Variable(torch.from_numpy(label))
#
#         out1, out2 = net.forward(x1, x2)
#         M_predict = data.calc_domination_matrix(F=out1.data.numpy(), _F=out2.data.numpy())
#         prec1 = (M_gt == M_predict).sum() / (M_predict.shape[0] * M_gt.shape[0])
#         M_gt = Variable((torch.from_numpy(M_gt)).float(), requires_grad=True)
#         M_predict = Variable((torch.from_numpy(M_predict)).float(), requires_grad=True)
#         loss = criterion(M_gt, M_predict)
#         losses.update(loss.data.item(), batch_size)
#         top1.update(prec1, batch_size)
#
#         print('Test Epoch: %d  | Loss: %.4f | Acc: %.4f ' % (epoch, losses.avg, top1.avg))
#
#     # return losses.avg, top1.avg.cpu().numpy()
#     return losses.avg, top1.avg
# # Test
# def test():
#     net.eval()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     dataiter = iter(testloader)
#
#     for batch_idx, (index1, x1, f1, g1) in enumerate(testloader):
#
#         index2, x2, f2, g2 = dataiter.next()
#         indices = np.vstack((index1.data.numpy(),index2.data.numpy())).transpose()
#         label = data.test_dominance_matrix[indices[:, 0], indices[:, 1]]
#
#         if use_cuda:
#             x1, x2, f1, f2, g1, g2 = Variable(x1.cuda()), Variable(x2.cuda()), Variable(f1.cuda()), Variable(f2.cuda()), Variable(g1.cuda()), Variable(g2.cuda())
#             label = Variable(label.cuda())
#         else:
#             x1, x2, f1, f2, g1, g2 = Variable(x1), Variable(x2), Variable(f1), Variable(f2), Variable(g1), Variable(g2)
#             label = Variable(torch.from_numpy(label))
#
#         output= net.forward(x1, x2)
#         loss = criterion(output, label)
#
#         # measure accuracy and record loss for learner
#         prec1 = accuracy(output.data, label.data)
#         losses.update(loss.data.item(), batch_size)
#         top1.update(prec1[0], batch_size)
#
#         print('Test Epoch: %d  | Loss: %.4f | Acc: %.4f ' % (epoch, losses.avg, top1.avg.cpu().numpy()))
#
#     return losses.avg, top1.avg.cpu().numpy()


for epoch in range(start_epoch, start_epoch + total_epoch):
    print('\nEpoch: %d' % epoch)
    scheduler1.step()
    train_loss, train_acc = train()
    test_loss, test_acc = test()
    logger.append([optimizer.param_groups[0]['lr'], float(train_loss), float(test_loss), float(train_acc), float(test_acc)])

    # Save checkpoint.
    if test_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'state_dict': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/' + checkpoint_filename)
        best_acc = test_acc


print("Train Data Labels: 0==(%f), 1==(%f), 2==(%f)"% (data.train_label_zero, data.train_label_one, data.train_label_two))
print("Test Data Labels: 0==(%f), 1==(%f), 2==(%f)"% (data.test_label_zero, data.test_label_one, data.test_label_two))

logger.close()
print("Done")

