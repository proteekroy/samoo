from frameworks.framework12 import *
from frameworks.framework_hybrid import *
from frameworks.framework_switching import *
from abc import abstractmethod


class Framework:

    def __init__(self,
                 framework_id=None,
                 framework_crossval=None,
                 problem=None,
                 algorithm=None,
                 curr_ref=None,
                 model_list=None,
                 *args,
                 **kwargs
                 ):
        self.framework_id = framework_id
        self.framework_crossval = framework_crossval
        self.problem = problem
        self.algorithm = algorithm
        self.curr_ref = curr_ref
        self.model_list = model_list
        super().__init__()

    @abstractmethod
    def train(self, x, f, g, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x, f, g, *args, **kwargs):
        pass


def get_framework(framework_id=None,
                  framework_crossval=None,
                  problem=None,
                  algorithm=None,
                  curr_ref=None,
                  model_list=None,
                  *args,
                  **kwargs):
    frameworks = framework_id.split(',')
    if len(frameworks)==1:
        if framework_id.lower() in ['12a']:
            return Framework12A(framework_id, framework_crossval, problem, algorithm, curr_ref, model_list, *args, **kwargs)
        elif framework_id.lower() in ['12b']:
            return Framework12B(framework_id, framework_crossval, problem, algorithm, curr_ref, model_list, *args, **kwargs)
        elif framework_id.lower() in ['12c']:
            return Framework12C(framework_id, framework_crossval, problem, algorithm, curr_ref, model_list, *args, **kwargs)
        elif framework_id.lower() in ['12d']:
            return Framework12D(framework_id, framework_crossval, problem, algorithm, curr_ref, model_list, *args, **kwargs)
        elif framework_id.lower() in ['hybrid']:
            return FrameworkHybrid(framework_id, framework_crossval, problem, algorithm, curr_ref, model_list, *args, **kwargs)
        else:
            raise Exception("Framework not supported")
    else:
        return FrameworkSwitching(framework_id, framework_crossval, problem, algorithm, curr_ref, model_list, *args, **kwargs)


