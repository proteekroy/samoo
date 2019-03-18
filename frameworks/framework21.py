from frameworks.factory import Framework
import numpy as np


class Framework21(Framework):
    def __init__(self,
                 framework_id=None,
                 problem=None,
                 algorithm=None,
                 model_list=None,
                 ref_dirs=None,
                 curr_ref_id=None,
                 f_aggregate_func='asf',
                 g_aggregate_func='acv',
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
                         g_aggregate_func=g_aggregate_func,
                         *args,
                         **kwargs)

    def train(self, x, f, g, *args, **kwargs):

        for i in range(0, self.problem.n_obj):
            self.model_list["f"+str(i+1)].train(x, f[:, i])

        if self.problem.n_constr > 0:
            out = dict()
            self.prepare_aggregate_data(x=x, f=f, g=g,
                                        g_aggregate=self.g_aggregate_func, out=out)
            self.model_list["G"].train(x, out['G'])

    def predict(self, x, out, *args, **kwargs):
        f = []
        g = []
        for i in range(0, self.problem.n_obj):
            _f = self.model_list["f"+str(i+1)].predict(x)
            f.append(_f)

        if self.problem.n_constr > 0:
            _g = self.model_list["G"].predict(x)
        else:
            _g = np.zeros(x.shape[0])

        g.append(_g)

        F = np.column_stack(f)
        G = np.column_stack(g)

        a_out = dict()
        Framework.prepare_aggregate_data(f=F,
                                         g=G,
                                         out=a_out,
                                         f_aggregate=self.f_aggregate_func,
                                         g_aggregate=None,
                                         m5_fg_aggregate=None,
                                         m6_fg_aggregate=None,
                                         ref_dirs=self.ref_dirs,
                                         curr_ref_id=self.curr_ref_id
                                         )

        out["F"] = a_out["F"]
        out["G"] = G
