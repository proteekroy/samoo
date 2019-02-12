from frameworks.factory import Framework
import numpy as np


class Framework42(Framework):
    def __init__(self,
                 framework_id=None,
                 problem=None,
                 algorithm=None,
                 model_list=None,
                 ref_dirs=None,
                 curr_ref_id=None,
                 g_aggregate_func='acv',
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
                         g_aggregate_func=g_aggregate_func,
                         f_aggregate_func=f_aggregate_func,
                         *args,
                         **kwargs)

    def train(self, x, f, g, *args, **kwargs):

        for i in range(len(self.ref_dirs)):
            out = dict()
            self.prepare_aggregate_data(x=x, f=f, g=g, out=out)
            self.model_list["f"+str(i+1)].train(x, out['F'])

        if self.problem.n_constr > 0:
            out = dict()
            self.prepare_aggregate_data(x=x, f=f, g=g, out=out)
            self.model_list["G"].train(x, out['G'])

    def predict(self, x, out, *args, **kwargs):
        f = []
        g = []
        for i in range(len(self.ref_dirs)):
            _f = self.model_list["f"+str(i+1)].predict(x)
            f.append(_f)

        if self.problem.n_constr > 0:
            _g = self.model_list["G"].predict(x)

        else:
            _g = np.zeros(x.shape[0])

        g.append(_g)

        out["F"] = np.column_stack(f)
        out["G"] = np.column_stack(g)
