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
                 metamodel_list=None,
                 acq_list=None,
                 framework_acq_dict=None,
                 aggregation=None,
                 n_split=None,
                 *args,
                 **kwargs
                 ):
        super().__init__()
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
        self.metamodel_list = metamodel_list
        self.acq_list = acq_list
        self.framework_acq_dict = framework_acq_dict
        self.aggregation = aggregation
        self.n_split = n_split
        self.type = -1
        self.sep = None

    @abstractmethod
    def train(self, x, f, g, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x, out, *args, **kwargs):
        pass

    def set_current_reference(self, curr_ref_id):
        self.curr_ref_id = curr_ref_id

    def constrained_domination(self, obj1, obj2, cv1, cv2):

        if (cv1 <= 0 and cv2 <= 0) or (cv1 - cv2) < 1e-16:
            return self.lex_dominate(obj1, obj2)
        else:
            if cv1 < cv2:
                return 1
            else:
                return 2

    def lex_dominate(self, obj1, obj2):
        first_dom_second = np.all(obj1 <= obj2) and np.any(obj1 < obj2)
        second_dom_first = np.all(obj2 <= obj1) and np.any(obj2 < obj1)
        if first_dom_second:
            return 1
        elif second_dom_first:
            return 2
        else:
            return 3