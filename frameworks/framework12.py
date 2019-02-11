from frameworks.factory import Framework
import copy
from metamodels.regression_metamodel import GPRmodel, SVRmodel, KRRmodel
from frameworks.normalize import NormalizeConstraint


class Framework12A(Framework):
    def __init__(self, problem, curr_ref, model_list, *args, ** kwargs):
        super().__init__(problem, curr_ref, model_list, *args, **kwargs)

        self.model_list["f1"] = self.model_list["f"]

        for i in range(1, self.problem.n_obj):
            self.model_list["f" + str(i + 1)] = copy.deepcopy(self.model_list["f"])

        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                self.model_list["g" + str(i + 1)] = copy.deepcopy(self.model_list["f"])

    def train(self, x, f, g, *args, **kwargs):
        for i in range(0, self.problem.n_obj):
            self.model_list["f"+str(i+1)].train(x, f[:, i])

        if self.problem.n_constr > 0:
            norm = NormalizeConstraint()
            g_normalized = g  # norm.normalize_constraint(g)
            for i in range(0, self.problem.n_constr):
                self.model_list["g"+str(i+1)].train(x, g_normalized[:, i])

    def predict(self, x, f, g, *args, **kwargs):
        for i in range(0, self.problem.n_obj):
            f[:, i] = self.model_list["f"+str(i+1)].predict(x)

        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                g[:, i] = self.model_list["g" + str(i + 1)].predict(x)


class Framework12B(Framework):
    def __init__(self, problem, curr_ref, model_list, *args, ** kwargs):
        super().__init__(problem, curr_ref=curr_ref,
                         model_list=model_list)

    def train(self, x, f, g, *args, **kwargs):

        self.model_list["f"].train(x, f, cross_val=False, *args, **kwargs)

        if self.problem.n_constr > 0:
            self.model_list["g"].train(x, g, cross_val=False, *args, **kwargs)

    def predict(self, x, f, g, *args, **kwargs):

        f[:, :] = self.model_list["f"].predict(x)

        if self.problem.n_constr > 0:
            g[:, :] = self.model_list["g"].predict(x)


class Framework12C(Framework):
    def __init__(self, problem, curr_ref, model_list, *args, ** kwargs):
        super().__init__(problem, curr_ref=curr_ref,
                         model_list=model_list)

        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                self.model_list["g" + str(i + 1)] = GPRmodel()

    def train(self, x, f, g, *args, **kwargs):

        self.model_list["f"].train(x, f, cross_val=False, *args, **kwargs)

        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                self.model_list["g"+str(i+1)].train(x, g[:, i])

    def predict(self, x, f, g, *args, **kwargs):

        f[:, :] = self.model_list["f"].predict(x)

        if self.problem.n_constr > 0:
            for i in range(0, self.problem.n_constr):
                g[:, i] = self.model_list["g" + str(i + 1)].predict(x)


class Framework12D(Framework):
    def __init__(self, problem, curr_ref, model_list, *args, ** kwargs):
        super().__init__(problem, curr_ref=curr_ref,
                         model_list=model_list)

        self.model_list["f1"] = GPRmodel()

        for i in range(1, self.problem.n_obj):
            self.model_list["f" + str(i + 1)] = GPRmodel()

    def train(self, x, f, g, *args, **kwargs):

        for i in range(0, self.problem.n_obj):
            self.model_list["f"+str(i+1)].train(x, f[:, i])

        if self.problem.n_constr > 0:
            self.model_list["g"].train(x, g, cross_val=False, *args, **kwargs)

    def predict(self, x, f, g, *args, **kwargs):

        for i in range(0, self.problem.n_obj):
            f[:, i] = self.model_list["f"+str(i+1)].predict(x)

        if self.problem.n_constr > 0:
            g[:, :] = self.model_list["g"].predict(x)
