from frameworks.factory import Framework
import numpy as np
from itertools import product
from metamodels.model_selection import prepare_data


class Framework6(Framework):
    def __init__(self,
                 framework_id=None,
                 problem=None,
                 algorithm=None,
                 model_list=None,
                 ref_dirs=None,
                 curr_ref_id=None,
                 m6_fg_aggregate_func='asfcv',
                 *args,
                 **kwargs
                 ):
        super().__init__(framework_id=framework_id,
                         problem=problem,
                         algorithm=algorithm,
                         model_list=model_list,
                         ref_dirs=ref_dirs,
                         curr_ref_id=curr_ref_id,
                         m6_fg_aggregate_func=m6_fg_aggregate_func,
                         *args,
                         **kwargs)
        self.type = 2

    def train(self, x, f, g, *args, **kwargs):

        for i in range(len(self.ref_dirs)):
            string = "fg_M6_" + str(i + 1) + "_" + str(self.m6_fg_aggregate_func)
            d = prepare_data(f=f, g=g, acq_func=[string], ref_dirs=self.ref_dirs,
                             curr_ref_id=i)

            self.model_list[string].train(x, d[string])

    def predict(self, x, out, *args, **kwargs):
        f = []
        g = []
        for i in range(len(self.ref_dirs)):
            _f = self.model_list["fg_M6_" + str(i + 1) + "_" + str(self.m6_fg_aggregate_func)].predict(x)
            f.append(_f)

        _g = np.zeros([x.shape[0], self.problem.n_constr])
        g.append(_g)

        out["F"] = np.column_stack(f)
        out["G"] = np.column_stack(g)

    def calculate_sep(self, problem, actual_data, prediction_data, n_split):

        err = []
        for partition in range(n_split):

            # compute average error over all reference directions
            temp_err = 0
            count = 0
            for j in range(len(self.ref_dirs)):
                string = "fg_M6_"+str(j + 1) + "_" + str(self.m6_fg_aggregate_func)
                I_temp = np.arange(0, actual_data[string][partition].shape[0])
                I = np.asarray(list(product(I_temp, I_temp)))
                cv = np.zeros([I_temp.shape[0], 1])
                cv_pred = np.zeros([I_temp.shape[0], 1])

                f = actual_data[string][partition]
                f_pred = prediction_data[string][partition]

                for i in range(I.shape[0]):
                    count = count + 1
                    d1 = self.constrained_domination(f[I[i, 0]], f[I[i, 1]], cv[I[i, 0]], cv[I[i, 1]])
                    d2 = self.constrained_domination(f_pred[I[i, 0]], f_pred[I[i, 1]], cv_pred[I[i, 0]], cv_pred[I[i, 1]])
                    if d1 != d2:
                        temp_err = temp_err + 1
            temp_err = temp_err/count
            err.append(temp_err)

        return np.asarray(err)
