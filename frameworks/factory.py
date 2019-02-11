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
