from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict
from itertools import product
from metamodels.siamese_net import SiameseMetamodel
from metamodels.nn_models import SiameseNet, ConstraintNet
from metamodels.constraint_net import ConstraintMetamodel
from metamodels.regression_metamodel import GPRmodel, SVRmodel, KRRmodel, DacefitGP


def data_partition(x, n_split):
    kf = KFold(n_splits=n_split)
    train = []
    test = []
    for train_index, test_index in kf.split(x):
        train.append(train_index)
        test.append(test_index)

    return train, test


def prepare_data(x=None,
                 f=None,
                 g=None,
                 acq_func=None,
                 ref_dirs=None,
                 curr_ref_id=0):

    d = {}
    for acq in acq_func:
        if 'f' in acq and 'l' not in acq and 'fg' not in acq and 'asf' not in acq:
            d[acq] = f[:, int(acq.split('f')[1])-1]
        elif 'g' in acq and 'fg' not in acq:
            d[acq] = g[:, int(acq.split('g')[1])-1]
        elif 'G' in acq:
            aggregation_function = acq.split('_')[1]
            cv = np.copy(g)
            index = np.any(g > 0, axis=1)
            cv[g <= 0] = 0
            cv = np.sum(cv, axis=1)

            if aggregation_function == 'cv':
                d[acq] = np.vstack(np.asarray(cv).flatten())
            elif aggregation_function == 'acv':
                acv = np.sum(g, axis=1)
                acv[index] = np.copy(cv[index])
                d[acq] = np.vstack(np.asarray(acv).flatten())
            else:
                raise Exception('Aggregation function for constraints not defined!')
            d[acq] = np.expand_dims(d[acq], axis=1)
        elif 'l' in acq:
            f_normalized = (f - np.min(f, axis=0)) / (np.max(f, axis=0) - np.min(f, axis=0))
            aggregation_function = acq.split('_')[1]
            if aggregation_function == 'asf_regular':  # directions pass through single ideal points
                d[acq] = np.max(f_normalized / ref_dirs[curr_ref_id], axis=1)
            elif aggregation_function == 'asf':  # directions pass through parallel lines, multiple ideal points
                d[acq] = np.max(f_normalized - ref_dirs[curr_ref_id], axis=1)
            else:
                raise Exception('Aggregation function for objectives not defined!')
            d[acq] = np.expand_dims(d[acq], axis=1)

        elif 'fg_M5' in acq:
            aggregation_function = acq.split('_')[3]
            if aggregation_function == 'asfcv':
                cv = np.copy(g)
                cv[g <= 0] = 0
                cv = np.sum(cv, axis=1)
                index = cv > 0
                f_normalized = (f - np.min(f, axis=0)) / (np.max(f, axis=0) - np.min(f, axis=0))
                f_temp = np.max(f_normalized - ref_dirs[curr_ref_id], axis=1)
                f_max = np.max(f_temp)
                if len(index) > 0:
                    f_temp[index] = f_max + cv[index]
                d[acq] = f_temp
            else:
                raise Exception('M5 Aggregation of objectives and constraints not defined!')
            d[acq] = np.expand_dims(d[acq], axis=1)
        elif 'fg_M6' in acq:
            aggregation_function = acq.split('_')[2]
            if aggregation_function == 'asfcv':
                t = []
                cv = np.copy(g)
                cv[g <= 0] = 0
                cv = np.sum(cv, axis=1)
                index = cv > 0
                for curr_ref_id in range(0, ref_dirs.shape[0]):
                    f_normalized = (f - np.min(f, axis=0)) / (np.max(f, axis=0) - np.min(f, axis=0))
                    f_temp = np.max(f_normalized - ref_dirs[curr_ref_id], axis=1)
                    f_max = np.max(f_temp)
                    if len(index) > 0:
                        f_temp[index] = f_max + cv[index]
                    t.append(f_temp)

                d[acq] = np.column_stack(t)
            elif aggregation_function == 'asf_regular_cv':
                t = []
                cv = np.copy(g)
                cv[g <= 0] = 0
                cv = np.sum(cv, axis=1)
                index = cv > 0

                for curr_ref_id in range(0, ref_dirs.shape[0]):
                    f_normalized = (f - np.min(f, axis=0)) / (np.max(f, axis=0) - np.min(f, axis=0))
                    f_temp = np.max(f_normalized / ref_dirs[curr_ref_id], axis=1)
                    f_max = np.max(f_temp)
                    if len(index) > 0:
                        f_temp[index] = f_max + cv[index]
                    t.append(f_temp)

                d[acq] = np.column_stack(t)

            elif aggregation_function == 'minasfcv':
                t = []
                cv = np.copy(g)
                cv[g <= 0] = 0
                cv = np.sum(cv, axis=1)
                index = cv > 0

                for curr_ref_id in range(0, ref_dirs.shape[0]):
                    f_normalized = (f - np.min(f, axis=0)) / (np.max(f, axis=0) - np.min(f, axis=0))
                    f_temp = np.max(f_normalized - ref_dirs[curr_ref_id], axis=1)
                    f_max = np.max(f_temp)
                    if len(index) > 0:
                        f_temp[index] = f_max + cv[index]
                    t.append(f_temp)

                t = np.column_stack(t)
                d[acq] = np.min(t, axis=1)
            else:
                raise Exception('M6 Aggregation of objectives and constraints not defined!')
    return d


def select_best_metamodel(x=None,
                          f=None,
                          g=None,
                          acq_func=None,
                          ref_dirs=None,
                          metamodel_list=None,
                          problem=None,
                          n_split=10):

    train_data, test_data = data_partition(x, n_split=n_split)

    model = {}
    error_data = defaultdict(list)
    prediction_data = defaultdict(list)
    actual_data = defaultdict(list)

    for acq in acq_func:  # for each acc function
        # print(acq)
        best_err = np.inf
        best_metamodel_name = metamodel_list[0]

        for name in metamodel_list:  # for each metamodel
            sum_err = 0
            temp_error_data = defaultdict(list)
            temp_prediction_data = defaultdict(list)
            temp_actual_data = defaultdict(list)
            for i in range(n_split):  # for each partition
                train_set = train_data[i]
                test_set = test_data[i]
                d_train = prepare_data(x=x[train_set], f=f[train_set], g=g[train_set], acq_func=[acq], ref_dirs=ref_dirs)
                d_test = prepare_data(x=x[test_set], f=f[test_set], g=g[test_set], acq_func=[acq], ref_dirs=ref_dirs)
                metamodel_i = return_metamodel_object(metamodel_name=name, problem=problem)
                metamodel_i.train(x[train_set], d_train[acq])
                err, f_pred = calculate_sep_metamodel(metamodel_i, x[test_set], np.expand_dims(d_test[acq], axis=1))
                sum_err = sum_err + err
                temp_error_data[acq].append(err)
                temp_prediction_data[acq].append(f_pred)
                temp_actual_data[acq].append(d_test[acq])
            if sum_err < best_err:
                best_err = sum_err
                best_metamodel_name = name
                prediction_data[acq] = temp_prediction_data[acq]
                actual_data[acq] = temp_actual_data[acq]
                error_data[acq] = temp_error_data[acq]

        metamodel = return_metamodel_object(metamodel_name=best_metamodel_name, problem=problem)
        # d = ModelSelection.prepare_data(x=x, f=f, g=g, acq_func=[acq], ref_dirs=ref_dirs)
        # metamodel.train(x, d[acq])
        model[acq] = metamodel

    return model, error_data, actual_data, prediction_data

def calculate_sep_metamodel(mm, x, f):

    I = np.arange(0, x.shape[0])
    I = np.asarray(list(product(I, I)))

    f_pred = mm.predict(x)

    actual_relation = f[I[:, 0]] <= f[I[:, 1]]
    pred_relation = f_pred[I[:, 0]] <= f_pred[I[:, 1]]

    err = np.sum(actual_relation != pred_relation) / I.shape[0]

    return err, f_pred

def return_metamodel_object(metamodel_name, problem):

    if metamodel_name == 'dacefit':
        return DacefitGP()
    elif metamodel_name == 'gpr':
        return GPRmodel()
    elif metamodel_name == 'krr':
        return KRRmodel()
    elif metamodel_name == 'svr':
        return SVRmodel()
    else:
        raise Exception('Metamodel not supported')
    # elif metamodel_name == 'siamese':
    #     net = SiameseNet(n_var=problem.n_var, n_obj=problem.n_obj, hidden_layer_length=20, embed_length=10)
    #     return SiameseMetamodel(problem.n_var, problem.n_obj, neuralnet=net,
    #                             total_epoch=300, disp=False,
    #                             best_accuracy_model=False,
    #                             batch_size=10)
    # elif metamodel_name == 'constrnet':
    #     net = ConstraintNet(n_var=problem.n_var, n_constr=problem.n_constr)
    #     return ConstraintMetamodel(problem.n_var, problem.n_constr,
    #                                neuralnet=net, total_epoch=300,
    #                                disp=False, best_accuracy_model=False)

