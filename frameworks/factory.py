from abc import abstractmethod
import numpy as np


class Framework:

    def __init__(self,
                 framework_id=None,
                 framework_crossval=None,
                 problem=None,
                 algorithm=None,
                 model_list=None,
                 ref_dirs=None,
                 curr_ref_id=None,
                 g_aggregate_func=None,
                 f_aggregate_func=None,
                 m5_fg_aggregate_func=None,
                 m6_fg_aggregate_func=None,
                 *args,
                 **kwargs
                 ):
        self.framework_id = framework_id
        self.framework_crossval = framework_crossval
        self.problem = problem
        self.algorithm = algorithm
        self.ref_dirs = ref_dirs
        self.curr_ref_id = curr_ref_id
        self.model_list = model_list
        self.g_aggregate_func = g_aggregate_func
        self.f_aggregate_func = f_aggregate_func
        self.m5_fg_aggregate_func = m5_fg_aggregate_func
        self.m6_fg_aggregate_func = m6_fg_aggregate_func
        self.output = dict()
        self.input = dict()
        super().__init__()

    @abstractmethod
    def train(self, x, f, g, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x, out, *args, **kwargs):
        pass

    def prepare_aggregate_data(self, x = None, f = None, g = None, out=None,  *args, ** kwargs):

        if self.f_aggregate_func is not None:
            if self.f_aggregate_func == 'asf_regular':
                F = np.max(f / self.ref_dirs[self.curr_ref_id], axis=1)
            elif self.f_aggregate_func == 'asf':
                F = np.max(f - self.ref_dirs[self.curr_ref_id], axis=1)
            else:
                raise Exception('Aggregation function for objectives not defined.')
            out['F'] = F

        if self.g_aggregate_func is not None:
            cv = np.copy(g)
            index = np.any(g > 0, axis=1)
            cv[g <= 0] = 0
            cv = np.sum(cv, axis=1)
            acv = np.sum(g, axis=1)
            acv[index] = np.copy(cv[index])

            if self.g_aggregate_func == 'cv':
                G = np.vstack(np.asarray(cv).flatten())
            elif self.g_aggregate_func == 'acv':
                G = np.vstack(np.asarray(acv).flatten())
            else:
                raise Exception('Aggregation function for constraints not defined.')

            out['G'] = G

        if self.m5_fg_aggregate_func is not None:
            F = np.max(f - self.ref_dirs[self.curr_ref_id], axis=1)
            cv = np.copy(g)
            cv[g <= 0] = 0
            cv = np.sum(cv, axis=1)
            out['S5'] = F+cv

        if self.m6_fg_aggregate_func is not None:
            F = np.zeros([f.shape[0], self.ref_dirs.shape[1]])
            for i in range(len(self.ref_dirs)):
                F[:, i] = np.max(f - self.ref_dirs[self.curr_ref_id], axis=1)

            F = np.min(F, axis=1)
            cv = np.copy(g)
            cv[g <= 0] = 0
            cv = np.sum(cv, axis=1)
            out['S6'] = F+cv
