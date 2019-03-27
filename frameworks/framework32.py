from frameworks.factory import Framework
import numpy as np


class Framework32(Framework):
    def __init__(self,
                 framework_id=None,
                 problem=None,
                 algorithm=None,
                 model_list=None,
                 ref_dirs=None,
                 curr_ref_id=None,
                 f_aggregate_func='asf',
                 *args,
                 **kwargs
                 ):
        super().__init__(framework_id=framework_id,
                         problem=problem,
                         algorithm=algorithm,
                         model_list=model_list,
                         ref_dirs=ref_dirs,
                         curr_ref_id=curr_ref_id,
                         f_aggregate_func=f_aggregate_func,
                         *args,
                         **kwargs)

    def train(self, x, f, g, *args, **kwargs):

        out = dict()
        for i in range(len(self.ref_dirs)):

            self.prepare_aggregate_data(x=x,
                                        f=f,
                                        g=g,
                                        out=out,
                                        f_aggregate='asf',
                                        ref_dirs=self.ref_dirs,
                                        curr_ref_id=i)
            self.model_list["f"+str(i+1)].train(x, out['F'])

        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                self.model_list["g" + str(i + 1)].train(x, g[:, i])

    def predict(self, x, out, *args, **kwargs):
        f = []
        g = []
        for i in range(len(self.ref_dirs)):
            _f = self.model_list["f"+str(i+1)].predict(x)
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
