from frameworks.factory import Framework
import numpy as np


class Framework5(Framework):
    def __init__(self,
                 framework_id=None,
                 problem=None,
                 algorithm=None,
                 model_list=None,
                 ref_dirs=None,
                 curr_ref_id=None,
                 m5_fg_aggregate_func='asfcv',
                 *args,
                 **kwargs
                 ):
        super().__init__(framework_id=framework_id,
                         problem=problem,
                         algorithm=algorithm,
                         model_list=model_list,
                         ref_dirs=ref_dirs,
                         curr_ref_id=curr_ref_id,
                         m5_fg_aggregate_func=m5_fg_aggregate_func,
                         *args,
                         **kwargs)

    def train(self, x, f, g, *args, **kwargs):

        out = dict()
        self.prepare_aggregate_data(x=x, f=f, g=g, out=out, ref_dirs=self.ref_dirs,
                                    m5_fg_aggregate=self.m5_fg_aggregate_func, curr_ref_id=self.curr_ref_id)
        self.model_list["f" + str(self.curr_ref_id + 1)].train(x, out['S5'])

    def predict(self, x, out, *args, **kwargs):
        f = []
        g = []
        _f = self.model_list["f" + str(self.curr_ref_id + 1)].predict(x)
        f.append(_f)
        _g = np.zeros(x.shape[0])
        g.append(_g)

        out["F"] = np.column_stack(f)
        out["G"] = np.column_stack(g)
