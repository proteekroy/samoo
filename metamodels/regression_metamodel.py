from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic, ExpSineSquared
import numpy as np
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from metamodels.neural_metamodel import MetaModel


class GPRmodel(MetaModel):

    def __init__(self):
        self.n_restarts_optimizer = 10
        self.alpha = 1e-10
        self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        super().__init__()

    def train(self, input, target, *args, **kwargs):
        # ker_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
        ker_rbf = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        ker_rq = ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(alpha=0.1, length_scale=1)
        # ker_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
        # kernel_list = [ker_rbf, ker_rq]
        kernel_list = [ker_rbf]

        param_grid = {"kernel": kernel_list,
                      "alpha": [1e-10, 1e-2, 1e-1, 1e1, 1e2],
                      "optimizer": ["fmin_l_bfgs_b"],
                      "n_restarts_optimizer": [10],
                      "normalize_y": [False],
                      "copy_X_train": [True],
                      "random_state": [0]}

        gp = GaussianProcessRegressor()
        self.model = GridSearchCV(gp, param_grid=param_grid)
        # grid_search.fit(X, y)
        # self.model = GridSearchCV(GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_restarts_optimizer, alpha=self.alpha), cv=5,
        #                           param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
        # self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_restarts_optimizer, alpha=self.alpha)
        self.model.fit(input, target)

    def predict(self, x, *args, **kwargs):
        mean = self.model.predict(x)
        return mean


class SVRmodel(MetaModel):

    def __init__(self):
        super().__init__()

    def train(self, input, target, *args, **kwargs):
        self.model = GridSearchCV(SVR(kernel='rbf', gamma='scale', epsilon=0.1, max_iter=1000, degree=2), cv=10,
                             param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
        self.model.fit(input, target)

    def predict(self, x, *args, **kwargs):
        out = self.model.predict(x)

        return out


class KRRmodel(MetaModel):

    def __init__(self):
        super().__init__()

    def train(self, input, target, *args, **kwargs):
        self.model = GridSearchCV(KernelRidge(kernel='rbf', alpha=0.01, degree=2), cv=10, param_grid={"gamma": np.logspace(-2, 2, 5)})
        self.model.fit(input, target)

    def predict(self, x, *args, **kwargs):
        out = self.model.predict(x)

        return out
