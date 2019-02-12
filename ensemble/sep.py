from ensemble.sep_framework import *
from ensemble.sep_metamodel import *
import numpy as np
from itertools import product
from sklearn.model_selection import KFold
from threading import Thread
from multiprocessing import Pool
import multiprocessing as mp
from frameworks.factory import Framework



def cross_val_framework(framework_list, metamodel_list, problem, x, f, g, n_splits=10):

    membership_set = np.isin(['11', '12', '21', '22', '31', '32', '41', '42', '5', '6'], framework_list)

    for fr in framework_list:
        if fr.framework_id.lower() in ['11', '12', '21', '22']:


    func_L = dict()

    if np.any(membership_set[0:4]):
        func_L['F'] = 'F'

    if membership_set[0] or membership_set[1] or membership_set[4] or membership_set[5]:
        func_L['g'] = 'g'

    if membership_set[2] or membership_set[3] or membership_set[6] or membership_set[7]:

        func_L['G'] =

    if membership_set[4]:
        func_L.append('S31')

    if membership_set[5]:
        func_L.append('S32')

    if membership_set[6]:
        func_L.append('S41')

    if membership_set[7]:
        func_L.append('S42')

    if membership_set[8]:
        func_L.append('S5')

    if membership_set[9]:
        func_L.append('S6')


    kf = KFold(n_splits=n_splits)
    data_L = []  # list of dictionaries for (framework, data) pair
    counter = 1
    processes = []
    for mm in metamodel_list:
        for train_index, test_index in kf.split(x):
            out = dict()
            data_L.append(out)
            if 'F' in func_L:
                for i in problem.n_obj:
                    p = mp.Process(target=sep_metamodel, args=[mm, f[train_index, i], f[test_index, i], out])
                    p.start()
                    processes.append(p)

            if 'g' in func_L:
                for j in problem.n_constr:
                    p = mp.Process(target=sep_metamodel, args=[mm, g[train_index, i], g[test_index, i], out])
                    p.start()
                    processes.append(p)

            if 'G' in func_L:
                target = Framework.prepare_data_M2_M4(g=g)
                p = mp.Process(target=sep_metamodel, args=[mm, target[train_index], out])
                p.start()
                processes.append(p)

            if fr.framework_id.lower() in ['21', '22']:
                target = Framw.prepare_data(g=g)
                requiredFuncList[counter]['G'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['31']:
                target = fr.prepare_data(f=f)
                requiredFuncList[counter]['S31'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['32']:
                target = fr.prepare_data(f=f)
                requiredFuncList[counter]['S32'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['41']:
                target = fr.prepare_data(f=f)
                requiredFuncList[counter]['S41'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['42']:
                target = fr.prepare_data(f=f)
                requiredFuncList[counter]['S42'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['5']:
                target = fr.prepare_data(f=f, g=g)
                requiredFuncList[counter]['S5'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['6']:
                target = fr.prepare_data(f=f, g=g)
                requiredFuncList[counter]['S6'] = list([target[train_index], target[test_index]])

            counter = counter + 1

    for process in processes:
        process.join()

    for out in data_L:
        print(out)

    dictList = []

    for i in range(5):
        out = dict()
        dictList.append(out)
        p = mp.Process(target=compute_sep_error, args=[metamodel, framework, train_data, test_data, dictList[i]])
        p.start()

    for i in range(5):
        p = mp.Process(target=compute_sep_error)
        p.start()
        p.join()
    threads = []
    # In this case 'urls' is a list of urls to be crawled.
    for ii in range(len(urls)):
        # We start one thread per url present.
        process = Thread(target=crawl, args=[urls[ii], result, ii])
        process.start()
        threads.append(process)
    # We now pause execution on the main thread by 'joining' all of our started threads.
    # This ensures that each has finished processing the urls.



def cross_val_parallel_framework(framework_list, problem, x, f, g, n_splits=10):
    kf = KFold(n_splits=n_splits)
    requiredFuncList = []  # list of dictionaries for (framework, data) pair
    counter = 1
    for fr in framework_list:
        for train_index, test_index in kf.split(x):
            out = dict()
            requiredFuncList.append(out)

            if fr.framework_id.lower() in ['11', '12', '21', '22']:
                for i in problem.n_obj:
                    requiredFuncList[counter]['f' + str(i + 1)] = list([f[train_index], f[test_index]])

            if fr.framework_id.lower() in ['11', '12']:
                for j in problem.n_constr:
                    requiredFuncList[counter]['g' + str(j + 1)] = list([f[train_index], f[test_index]])

            if fr.framework_id.lower() in ['21', '22']:
                target = fr.prepare_data(g=g)
                requiredFuncList[counter]['G'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['31']:
                target = fr.prepare_data(f=f)
                requiredFuncList[counter]['S31'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['32']:
                target = fr.prepare_data(f=f)
                requiredFuncList[counter]['S32'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['41']:
                target = fr.prepare_data(f=f)
                requiredFuncList[counter]['S41'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['42']:
                target = fr.prepare_data(f=f)
                requiredFuncList[counter]['S42'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['5']:
                target = fr.prepare_data(f=f, g=g)
                requiredFuncList[counter]['S5'] = list([target[train_index], target[test_index]])

            if fr.framework_id.lower() in ['6']:
                target = fr.prepare_data(f=f, g=g)
                requiredFuncList[counter]['S6'] = list([target[train_index], target[test_index]])

            counter = counter + 1



def sep_metamodel(metamodel, train_data, test_data, out):
    # Keep everything in try/catch loop so we handle errors

    for
    metamodel.train(train_data['x'])
    self.model_list["f1"]

    x  = train_data['x']


def sep_framework(framework_list, x, f, g):  # selection error probability for framework

    sep_array = []
    f_pred = np.zeros([, 1])
    for fr in framework_list:

        f, g = fr.predict(x, f, g)
        calculate_sep_framework(f,)
        # if fr.framework_id.lower() in ['11']:
        #     sep = sep_framework1(framework_id=11, f=f, g=g)
        # elif fr.framework_id.lower() in ['12']:
        #     sep = sep_framework2(framework_id=12, f=f, g=g)
        # elif fr.framework_id.lower() in ['21']:
        #     sep = sep_framework2(framework_id=21, f=f, g=g)
        # elif fr.framework_id.lower() in ['22']:
        #     sep = sep_framework2(framework_id=22, f=f, g=g)
        # elif fr.framework_id.lower() in ['31']:
        #     sep = sep_framework3(framework_id=31, f=f, g=g)
        # elif fr.framework_id.lower() in ['32']:
        #     sep = sep_framework3(framework_id=32, f=f, g=g)
        # elif fr.framework_id.lower() in ['41']:
        #     sep = sep_framework4(framework_id=41, f=f, g=g)
        # elif fr.framework_id.lower() in ['42']:
        #     sep = sep_framework4(framework_id=42, f=f, g=g)
        # elif fr.framework_id.lower() in ['5']:
        #     sep = sep_framework5(f=f, g=g)
        # elif fr.framework_id.lower() in ['6']:
        #     sep = sep_framework6(f=f, g=g)
        # else:
        #     raise Exception("Framework not supported for SEP calculation.")

        sep_array.append(sep)

    return np.asarray(sep_array).astype(float)


# def sep_metamodel(metamodel_list, f, g):  # selection error probability for framework
#
#     sep_array = []
#
#     for mm in metamodel_list:
#             sep = calculate_sep_metamodel(mm, x, f)
#
#         sep_array.append(sep)
#
#     return np.asarray(sep_array).astype(float)



def calculate_sep_metamodel(mm, x, f):

    I = np.arange(0, x.shape[0])
    I = np.asarray(list(product(I, I)))

    f_pred = mm.predict(x)

    actual_relation = (np.sum(f[I[:, 0]] <= f[I[:, 1]]))
    pred_relation = (np.sum(f_pred[I[:, 0]] <= f_pred[I[:, 1]]))

    err = np.sum(actual_relation==pred_relation)/(I.shape[0]*I.shape[1])

    return err


def calculate_sep_framework(f, f_pred, g, g_pred):
    cv = g
    cv[cv <=0] = 0
    cv = np.sum(cv, axis=1)
    f_max = np.max(f)
    I = g<=0
    f[I] = f_max + cv
    cv_pred = g_pred
    cv_pred[cv_pred <= 0] = 0
    cv_pred = np.sum(cv_pred, axis=1)
    f_max = np.max(f_pred)
    I = g_pred<=0
    f_pred[I] = f_max + cv_pred
    I = np.arange(0, f.shape[0])
    I = np.asarray(list(product(I, I)))

    actual_relation = (np.sum(f[I[:, 0]] <= f[I[:, 1]]))
    pred_relation = (np.sum(f_pred[I[:, 0]] <= f_pred[I[:, 1]]))

    err = np.sum(actual_relation==pred_relation)/(I.shape[0]*I.shape[1])

    return err