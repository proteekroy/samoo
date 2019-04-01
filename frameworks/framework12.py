from frameworks.factory import Framework
import numpy as np
from itertools import product


class Framework12(Framework):
    def __init__(self,
                 framework_id=None,
                 problem=None,
                 algorithm=None,
                 ref_dirs=None,
                 model_list=None,
                 curr_ref_id=None,
                 *args,
                 **kwargs
                 ):
        super().__init__(framework_id=framework_id,
                         problem=problem,
                         algorithm=algorithm,
                         model_list=model_list,
                         ref_dirs=ref_dirs,
                         curr_ref_id=curr_ref_id,
                         *args,
                         **kwargs)
        self.type = 2

    def train(self, x, f, g, *args, **kwargs):
        for i in range(0, self.problem.n_obj):
            self.model_list["f"+str(i+1)].train(x, f[:, i])

        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                self.model_list["g"+str(i+1)].train(x, g[:, i])

    def predict(self, x, out, *args, **kwargs):
        f = []
        g = []
        for i in range(0, self.problem.n_obj):
            _f = self.model_list["f" + str(i + 1)].predict(x)
            f.append(_f)
        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                _g = self.model_list["g" + str(i + 1)].predict(x)
                g.append(_g)
        else:
            _g = np.zeros(x.shape[0])
            g.append(_g)

        out["F"] = np.column_stack(f)
        out["G"] = np.column_stack(g)

    def calculate_sep(self, problem, actual_data, prediction_data, n_split):

        err = []
        for partition in range(n_split):
            f = []
            f_pred = []
            for i in range(problem.n_obj):
                f.append(actual_data['f'+str(i+1)][partition])
                f_pred.append(prediction_data['f' + str(i + 1)][partition])

            f = np.column_stack(f)
            f_pred = np.column_stack(f_pred)
            if problem.n_constr > 0:
                g = []
                g_pred = []
                for j in range(problem.n_constr):
                    g.append(actual_data['g' + str(j + 1)][partition])
                    g_pred.append(prediction_data['g' + str(j + 1)][partition])

                g = np.column_stack(g)
                g_pred = np.column_stack(g_pred)

                cv = np.copy(g)
                cv[g <= 0] = 0
                cv = np.sum(cv, axis=1)

                cv_pred = np.copy(g_pred)
                cv_pred[g_pred <= 0] = 0
                cv_pred = np.sum(cv_pred, axis=1)
            else:
                cv = np.zeros([f.shape[0], 1])
                cv_pred = np.zeros([f.shape[0], 1])

            I = np.arange(0, f.shape[0])
            I = np.asarray(list(product(I, I)))
            temp_err = 0
            for i in range(I.shape[0]):
                d1 = self.constrained_domination(f[I[i, 0]], f[I[i, 1]], cv[I[i, 0]], cv[I[i, 1]])
                d2 = self.constrained_domination(f_pred[I[i, 0]], f_pred[I[i, 1]], cv_pred[I[i, 0]], cv_pred[I[i, 1]])
                if d1 != d2:
                    temp_err = temp_err + 1
            err.append(temp_err)
        return np.asarray(err)
