from frameworks.factory import Framework
import copy
from metamodels.regression_metamodel import GPRmodel, SVRmodel, KRRmodel
import numpy as np


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


# class Framework12B(Framework):
#     def __init__(self,
#                  framework_id=None,
#                  framework_crossval=None,
#                  problem=None,
#                  algorithm=None,
#                  curr_ref=None,
#                  model_list=None,
#                  *args,
#                  **kwargs
#                  ):
#         super().__init__(framework_id, framework_crossval, problem, algorithm, curr_ref, model_list, *args, **kwargs)
#
#     def train(self, x, f, g, *args, **kwargs):
#
#         self.model_list["f"].train(x, f, cross_val=False, *args, **kwargs)
#
#         if self.problem.n_constr > 0:
#             self.model_list["g"].train(x, g, cross_val=False, *args, **kwargs)
#
#     def predict(self, x, f, g, *args, **kwargs):
#
#         f[:, :] = self.model_list["f"].predict(x)
#
#         if self.problem.n_constr > 0:
#             g[:, :] = self.model_list["g"].predict(x)
#
#
# class Framework12C(Framework):
#     def __init__(self,
#                  framework_id=None,
#                  framework_crossval=None,
#                  problem=None,
#                  algorithm=None,
#                  curr_ref=None,
#                  model_list=None,
#                  *args,
#                  **kwargs
#                  ):
#         super().__init__(framework_id, framework_crossval, problem, algorithm, curr_ref, model_list, *args, **kwargs)
#
#         if self.problem.n_constr > 0:
#             for i in range(0, self.problem.n_constr):
#                 self.model_list["g" + str(i + 1)] = GPRmodel()
#
#     def train(self, x, f, g, *args, **kwargs):
#
#         self.model_list["f"].train(x, f, cross_val=False, *args, **kwargs)
#
#         if self.problem.n_constr > 0:
#             for i in range(0, self.problem.n_constr):
#                 self.model_list["g"+str(i+1)].train(x, g[:, i])
#
#     def predict(self, x, f, g, *args, **kwargs):
#
#         f[:, :] = self.model_list["f"].predict(x)
#
#         if self.problem.n_constr > 0:
#             for i in range(0, self.problem.n_constr):
#                 g[:, i] = self.model_list["g" + str(i + 1)].predict(x)
#
#
# class Framework12D(Framework):
#     def __init__(self,
#                  framework_id=None,
#                  framework_crossval=None,
#                  problem=None,
#                  algorithm=None,
#                  curr_ref=None,
#                  model_list=None,
#                  *args,
#                  **kwargs
#                  ):
#         super().__init__(framework_id, framework_crossval, problem, algorithm, curr_ref, model_list, *args, **kwargs)
#
#         self.model_list["f1"] = GPRmodel()
#
#         for i in range(1, self.problem.n_obj):
#             self.model_list["f" + str(i + 1)] = GPRmodel()
#
#     def train(self, x, f, g, *args, **kwargs):
#
#         for i in range(0, self.problem.n_obj):
#             self.model_list["f"+str(i+1)].train(x, f[:, i])
#
#         if self.problem.n_constr > 0:
#             self.model_list["g"].train(x, g, cross_val=False, *args, **kwargs)
#
#     def predict(self, x, f, g, *args, **kwargs):
#
#         for i in range(0, self.problem.n_obj):
#             f[:, i] = self.model_list["f"+str(i+1)].predict(x)
#
#         if self.problem.n_constr > 0:
#             g[:, :] = self.model_list["g"].predict(x)