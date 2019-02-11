from frameworks.factory import Framework
import copy
from metamodels.regression_metamodel import GPRmodel, SVRmodel, KRRmodel
from frameworks.normalize import NormalizeConstraint


class FrameworkHybrid(Framework):
    def __init__(self, problem, curr_ref, model_list, *args, ** kwargs):
        super().__init__(problem, curr_ref=curr_ref,
                         model_list=model_list)

        self.model_list["z1"] = self.model_list["gpr"]

        for i in range(1, self.problem.n_obj):
            self.model_list["z" + str(i + 1)] = copy.deepcopy(self.model_list["gpr"])

        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                self.model_list["g" + str(i + 1)] = GPRmodel()

    def train(self, x, f, g, *args, **kwargs):

        self.model_list["f"].train(x, f, cross_val=False, *args, **kwargs)
        z = self.model_list["f"].predict(x)  # obtain representation

        for i in range(0, self.problem.n_obj):
            self.model_list["z"+str(i+1)].train(z, f[:, i])

        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                self.model_list["g"+str(i+1)].train(x, g[:, i])

    def predict(self, x, f, g, *args, **kwargs):

        z = self.model_list["f"].predict(x)
        for i in range(0, self.problem.n_obj):
            f[:, i] = self.model_list["z"+str(i+1)].predict(z)

        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                g[:, i] = self.model_list["g" + str(i + 1)].predict(x)