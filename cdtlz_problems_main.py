import numpy as np
from pymop.problems import Problem
from pymop.problems.dtlz import DTLZ, DTLZ1, DTLZ2, DTLZ3, DTLZ4, generic_sphere

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import math
import seaborn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# from colorspacious import cspace_converter
from collections import OrderedDict
import matplotlib.colors as colors
import pandas as pd
import autograd.numpy as anp
from pymoo.util.reference_direction import UniformReferenceDirectionFactory


class C1(Problem):  # infeasible band

    def __init__(self, dtlz, r=None):
        super().__init__(dtlz.n_var, dtlz.n_obj, 1, dtlz.xl, dtlz.xu)
        self.dtlz = dtlz

        n_obj = dtlz.n_obj
        if r is None:
            if n_obj == 2:
                r = 6
            elif n_obj == 3:
                r = 6
            elif n_obj == 5:
                r = 12.5
            elif n_obj == 8:
                r = 12.5
            elif n_obj == 10:
                r = 15
            elif n_obj == 15:
                r = 9
            else:
                raise Exception("Unknown default r for problem.")

        self.r = r

    def _evaluate(self, X, out, *args, **kwargs):

        # merge the result we get from the dtlz
        self.dtlz._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c1(out["F"], self.r)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        return self.dtlz.pareto_front(ref_dirs, *args, **kwargs)


class C2(Problem):

    def __init__(self, dtlz, r=None):
        super().__init__(dtlz.n_var, dtlz.n_obj, 1, dtlz.xl, dtlz.xu)
        self.dtlz = dtlz

        n_obj = dtlz.n_obj
        if r is None:
            if n_obj == 2:
                r = 0.2
            elif n_obj == 3:
                r = 0.4
            else:
                r = 0.5

        self.r = r

    def _evaluate(self, X, out, *args, **kwargs):

        # merge the result we get from the dtlz
        self.dtlz._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c2(out["F"], self.r)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):

        F = self.dtlz.pareto_front(ref_dirs, *args, **kwargs)
        G = constraint_c2(F, r=self.r)
        G[G <= 0] = 0
        if G.ndim > 1:
            G = np.sum(G, axis=1)
        return F[G <= 0]


class C3(Problem):

    def __init__(self, dtlz,r=None):
        super().__init__(dtlz.n_var, dtlz.n_obj, 1, dtlz.xl, dtlz.xu)
        self.dtlz = dtlz

        n_obj = dtlz.n_obj
        if isinstance(dtlz, DTLZ1):
            self.dtlz_type = 1
        elif isinstance(dtlz, DTLZ2) or isinstance(dtlz, DTLZ3) or isinstance(dtlz, DTLZ4):
            self.dtlz_type = 2
        else:
            raise Exception("Unknown default r for problem.")

    def _evaluate(self, X, out, *args, **kwargs):

        # merge the result we get from the dtlz
        self.dtlz._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c3(out["F"], self.dtlz_type)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):

        F = self.dtlz.pareto_front(ref_dirs, *args, **kwargs)

        a = np.sqrt(np.sum(F ** 2, 1) - 3 / 4 * np.max(F ** 2, axis=1))
        a = np.expand_dims(a, axis=1)
        a = np.tile(a, [1, ref_dirs.shape[1]])
        F = F/a
        # F = F / np.tile((np.sqrt(np.sum(F**2, 2) - 3 / 4 * np.max(F**2, axis=1))), [1, ref_dirs.shape[0]])
        return F


class C4(Problem):

    def __init__(self, dtlz, r=None):
        super().__init__(dtlz.n_var, dtlz.n_obj, 1, dtlz.xl, dtlz.xu)
        self.dtlz = dtlz
        n_obj = dtlz.n_obj

        if r is None:
            if n_obj == 2:
                r = 0.225
            elif n_obj == 3:
                r = 0.225
            elif n_obj == 5:
                r = 0.225
            elif n_obj == 8:
                r = 0.26
            elif n_obj == 10:
                r = 0.26
            elif n_obj == 15:
                r = 0.27
            else:
                raise Exception("Parameter r is not defined ")
        self.r = r

    def _evaluate(self, X, out, *args, **kwargs):

        # merge the result we get from the dtlz
        self.dtlz._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c4(out["F"], self.r)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):

        F = self.dtlz.pareto_front(ref_dirs, *args, **kwargs)
        G = constraint_c4(F, r=self.r)
        G[G <= 0] = 0
        if G.ndim > 1:
            G = np.sum(G, axis=1)
        return F[G <= 0]


def constraint_c1(f, r, dtlz_type):

    if dtlz_type == 3:
        radius = np.sum(f ** 2, axis=1)
        g = - (radius - 16) * (radius - r ** 2)
    elif dtlz_type == 1:
        g = - (1 - f[:, -1] / 0.6 - np.sum(f[:, :-1] / 0.5, axis=1))
    return g


def constraint_c2(f, r):
    n_obj = f.shape[1]

    v1 = np.inf*np.ones(f.shape[0])

    for i in range(n_obj):
        temp = (f[:, i] - 1)**2 + (np.sum(f**2, axis=1)-f[:, i]**2) - r**2
        v1 = np.minimum(temp.flatten(), v1)

    a = 1/np.sqrt(n_obj)
    v2 = np.sum((f-a)**2, axis=1)-r**2
    g = np.minimum(v1, v2.flatten())

    return g


def constraint_c3(f, dtlz_type):  # M circle if DTLZ2,3,4 and linear otherwise
    n_obj = f.shape[1]
    g = np.zeros(f.shape)

    for i in range(n_obj):
        if dtlz_type == 1:
            g[:, i] = 1 - f[:, i] / 0.5 - (np.sum(f, axis=1) - f[:, i])
        else:
            g[:, i] = 1 - f[:, i] ** 2 / 4 - (np.sum(f ** 2, axis=1) - f[:, i] ** 2)

    return g


def constraint_c4(f, r):  # cylindrical
    l = np.mean(f, axis=1)
    l = np.expand_dims(l, axis=1)
    g = -np.sum(np.power(f-l, 2), axis=1) + np.power(r, 2)

    return g


class CDTLZ(DTLZ):

    def __init__(self, dtlz, clist=None, rlist=None, **kwargs):

        assert len(clist) == len(rlist)
        self.rlist = np.asarray(rlist)
        self.clist = np.asarray(clist)
        self.dtlz = dtlz

        super().__init__(n_var=dtlz.n_var, n_obj=dtlz.n_obj, **kwargs)
        self.n_constr = len(np.unique(clist))

        if isinstance(dtlz, DTLZ1):
            self.dtlz_type = 1
        elif isinstance(dtlz, DTLZ2):
            self.dtlz_type = 2
        elif isinstance(dtlz, DTLZ3):
            self.dtlz_type = 3
        elif isinstance(dtlz, DTLZ4):
            self.dtlz_type = 4
        else:
            raise Exception("DTLZ problem not supported.")

    def _evaluate(self, X, out, *args, **kwargs):

        # merge the result we get from the dtlz
        self.dtlz._evaluate(X, out, *args, **kwargs)

        g = []
        for i in range(self.n_constr):
            if clist[i] == 1:
                _g = constraint_c1(out["F"], self.rlist[i], self.dtlz_type)
            elif clist[i] == 2:
                _g = constraint_c2(out["F"], self.rlist[i])
            elif clist[i] == 3:
                _g = constraint_c3(out["F"], self.dtlz_type)
            elif clist[i] == 4:
                _g = constraint_c4(out["F"], self.rlist[i])
            else:
                _g = []

            g.append(_g)

        out["G"] = np.column_stack(g)

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):

        F = self.dtlz.pareto_front(ref_dirs, *args, **kwargs)

        if np.any(self.clist == 3):
            a = np.sqrt(np.sum(F ** 2, 1) - 3 / 4 * np.max(F ** 2, axis=1))
            a = np.expand_dims(a, axis=1)
            a = np.tile(a, [1, ref_dirs.shape[1]])
            F = F / a
        elif np.any(self.clist == 2):
            r = self.rlist[np.where(self.clist == 2)][0]
            _g = constraint_c2(F, r)
            _g[_g <= 0] = 0
            if _g.ndim > 1:
                _g = np.sum(G, axis=1)
            F = F[_g <= 0]
        elif np.any(self.clist == 4):
            r = self.rlist[np.where(self.clist == 4)][0]
            _g = constraint_c4(F, r)
            _g[_g <= 0] = 0
            if _g.ndim > 1:
                _g = np.sum(G, axis=1)
            F = F[_g <= 0]

        return F

#
#
# class C2DTLZ2(CDTLZ):
#
#     def __init__(self, n_var, n_obj, r = None, c=1):
#         # if M == 3; r = 0.4; else r = 0.5; end
#         if r is None:
#             if n_obj == 2:
#                 self.r = 0.2
#             elif n_obj == 3:
#                 self.r = 0.4
#             else:
#                 self.r = 0.5
#
#         self.constr_func = dict()
#         for i in list(c):
#             if i == 1:
#                 self.constr_func[str(i+1)] = constraint_c1
#             elif i == 2:
#                 self.constr_func[str(i+1)] = constraint_c2
#             elif i == 3:
#                 self.constr_func[str(i+1)] = constraint_c3
#             elif i == 4:
#                 self.constr_func[str(i+1)] = constraint_c4
#
#         super().__init__(n_var=n_var, n_obj=n_obj, n_constr=1)
#
#     def _calc_pareto_front(self, ref_dirs):
#         return generic_sphere(ref_dirs)
#
#     def _evaluate(self, x, f, g, *args, **kwargs):
#         X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
#         param_g = self.g1(X_M)
#         for i in range(0, self.n_obj):
#             f[:, i] = 0.5 * (1 + param_g)
#             f[:, i] *= np.prod(X_[:, :X_.shape[1] - i], axis=1)
#             if i > 0:
#                 f[:, i] *= 1 - X_[:, X_.shape[1] - i]
#
#         for i in len(self.constr_func):
#             g[:, i] = self.constr_func[str(i+1)](x, f, r=self.r)
#
#
# class C1DTLZ3(CDTLZ):
#
#     def __init__(self, n_var, n_obj, r = None, k=None):
#
#         if r is None:
#             if n_obj == 2:
#                 self.r = 6
#             elif n_obj == 3:
#                 self.r = 6
#             elif n_obj == 5:
#                 self.r = 12.5
#             elif n_obj == 8:
#                 self.r = 12.5
#             elif n_obj == 10:
#                 self.r = 15
#             elif n_obj == 15:
#                 self.r = 9
#             else:
#                 self.r = 4
#
#         super().__init__(n_var=n_var, n_obj=n_obj, n_constr=1)
#
#     def _calc_pareto_front(self, ref_dirs):
#         return generic_sphere(ref_dirs)
#
#     def _evaluate(self, x, f, g, *args, **kwargs):
#         X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
#         param_g = self.g1(X_M)
#         self.obj_func(X_, param_g, f, alpha=1)
#         radius = np.sum(f**2, axis=1)
#         constraint_c1(f, r=self.r)


def plot_test_problem(axarr, F, problem, type=1):

    F1 = F[:, 0].flatten()
    F2 = F[:, 1].flatten()

    c = list([1, 4, 3])
    constr_func = dict()
    for i in c:
        if i == 1:
            constr_func[str(i)] = constraint_c1
        elif i == 2:
            constr_func[str(i)] = constraint_c2
        elif i == 3:
            constr_func[str(i)] = constraint_c3
        elif i == 4:
            constr_func[str(i)] = constraint_c4
    # plt.clf()
    index= np.ones(F.shape[0], dtype=bool)

    for i in range(len(constr_func)):
        G = constr_func[str(c[i])](F, dtlz_type=type)
        cv = np.copy(G) # np.zeros(G.shape)

        light_grey = np.array([220,220,220])/256
        dark_grey = np.array([169, 169, 169])/256
        cv[G <= 0] = 0
        if G.ndim>1:
            cv = np.sum(cv, axis=1)

        temp_index = cv <= 0
        index = index & temp_index
        # axarr[index].plot(x, y)
        # cmaps = OrderedDict()
    plt.plot(F1, F2, 'o', color='#DCDCDC') #'#A9A9A9'
    plt.plot(F1[index], F2[index], 'o', color='b')
        # F_all = np.concatenate((F, np.vstack(cv)),axis=1)
        # heatmap, xedges, yedges = np.histogram2d(F1, F2, weights=cv, )
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        # plt.hexbin(F1[index], F2[index], C=cv, cmap='Greys')

        # ax.clf()
        # plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='Greys')
    ref_dirs = UniformReferenceDirectionFactory(n_dim=2, n_points=21)._do()
    PF = problem._calc_pareto_front(ref_dirs)
    plt.plot(PF[:, 0], PF[:, 1], 'o', color='r')
    plt.show()

        # df2 = pd.DataFrame(F_all)
        # df.to_csv('a.csv', index=False)
        # df2 = pd.read_csv('a.csv')
        # df3 = df2.pivot()
        # seaborn.heatmap(df3)
        # seaborn.heatmap()
        # plt.show()

# DTLZ1, r = 20
# problems = [('DTLZ1', [2, 2]), ('DTLZ2', [2, 2]), ('DTLZ3', [2, 2]), ('DTLZ4', [2, 2])]
# problems = [('DTLZ1', [2, 2])]
problems = [('CDTLZ', [DTLZ3(n_var=2, n_obj=2), [1, 3], [6, 0]])]
# clist = [1, 2, 4]
# rlist  = [6, .2, .225]
clist = [1, 3]
rlist  = [6, 0]


if __name__ == '__main__':

    fig, ax = plt.subplots(len(problems))
    # fig, ax = plt.subplots()
    # fig.suptitle('Sharing X axis')

    # plt.show()
    # test(ax, 0, type=1)
    option = 1
    count = 0
    for prob_no, entry in enumerate(problems):
        name, params = entry
        print("Testing: " + name)

        # X, F, CV = load(name)
        problem = globals()[name](*params)
        # problem = CDTLZ(dtlz_problem, clist=clist, rlist=rlist)

        if isinstance(problem.dtlz, DTLZ1):
            dtlz_type = 1
        elif isinstance(problem.dtlz, DTLZ2):
            dtlz_type = 2
        elif isinstance(problem.dtlz, DTLZ3):
            dtlz_type = 3
        elif isinstance(problem.dtlz, DTLZ4):
            dtlz_type = 4
        else:
            raise Exception("DTLZ problem not supported.")

        nx = 100
        if option == 1:
            x = np.linspace(0, 1, nx)

            if dtlz_type == 1:
                # y = np.linspace(0.5 - 7.5 * 1e-2, 0.5 + 7.5 * 1e-2, nx)  # DTLZ1
                y = np.linspace(0.5 - 7.5 * 1e-4, 0.5 + 7.5 * 1e-4, nx)  # DTLZ1
            elif dtlz_type == 2:
                y = np.linspace(0.5 - 0.2, 0.5 + 0.2, nx)
            elif dtlz_type == 3:
                y = np.linspace(0.5 - 7.5 * 1e-3, 0.5 + 7.5 * 1e-3, nx)  # DTLZ3
            elif dtlz_type == 4:
                y = np.linspace(0.5 - 0.4, 0.5 + 0.4, nx)

            # y = np.linspace(0, 1, nx)
            xv, yv = np.meshgrid(x, y)
            x = np.vstack(xv.flatten())
            y = np.vstack(yv.flatten())
            X = np.hstack((x, y))
            n = x.shape[0]

            out = dict()
            G = np.zeros((n, problem.n_constr))
            problem._evaluate(X, out)
            F = out["F"]
            G = out["G"]
        else:
            f1 = np.linspace(0, 10, nx)
            f2 = np.linspace(0, 10, nx)
            xv, yv = np.meshgrid(f1, f2)
            f1 = np.vstack(xv.flatten())
            f2 = np.vstack(yv.flatten())
            F = np.hstack((f1, f2))
            n = F.shape[0]

        # plot_test_problem(F, problem, type=dtlz_type)

        F1 = F[:, 0].flatten()
        F2 = F[:, 1].flatten()

        G[G <= 0] = 0
        if G.ndim > 1:
            G = np.sum(G, axis=1)
        index = G > 0

        ref_dirs = UniformReferenceDirectionFactory(n_dim=problem.n_obj, n_points=21)._do()
        PF = problem._calc_pareto_front(ref_dirs)

        plt.title(name)
        plt.plot(F1, F2, 'o', color='#DCDCDC')  # '#A9A9A9'
        plt.plot(F1[index], F2[index], 'o', color='b')
        plt.plot(PF[:, 0], PF[:, 1], 'o', color='r')
        # ax[count].set_title(name)
        # ax[count].plot(F1, F2, 'o', color='#DCDCDC')  # '#A9A9A9'
        # ax[count].plot(F1[index], F2[index], 'o', color='b')
        # ax[count].plot(PF[:, 0], PF[:, 1], 'o', color='r')
        # ax[count].set_xlim([np.min(F1), np.max(F1)])
        count += 1
    plt.show()


# if __name__ == '__main__':
#
#     fig, ax = plt.subplots(len(problems)*len(c_problems))
#     # fig, ax = plt.subplots()
#     # fig.suptitle('Sharing X axis')
#
#     # plt.show()
#     # test(ax, 0, type=1)
#     option = 1
#     count = 0
#     for prob_no, entry in enumerate(problems):
#         name, params = entry
#         print("Testing: " + name)
#
#         # X, F, CV = load(name)
#         dtlz_problem = globals()[name](*params)
#
#         for c_prob_no, entry in enumerate(c_problems):
#             C = globals()[entry]
#
#             problem = C(dtlz_problem, r=20)
#
#             if isinstance(problem.dtlz, DTLZ1):
#                 dtlz_type = 1
#             elif isinstance(problem.dtlz, DTLZ2):
#                 dtlz_type = 2
#             elif isinstance(problem.dtlz, DTLZ3):
#                 dtlz_type = 3
#             elif isinstance(problem.dtlz, DTLZ4):
#                 dtlz_type = 4
#             else:
#                 raise Exception("DTLZ problem not supported.")
#
#             nx = 100
#             if option == 1:
#                 x = np.linspace(0, 1, nx)
#
#                 if dtlz_type == 1:
#                     y = np.linspace(0.5 - 7.5 * 1e-2, 0.5 + 7.5 * 1e-2, nx)  # DTLZ1
#                 elif dtlz_type == 2:
#                     y = np.linspace(0.5 - 0.2, 0.5 + 0.2, nx)
#                 elif dtlz_type == 3:
#                     y = np.linspace(0.5 - 7.5 * 1e-3, 0.5 + 7.5 * 1e-3, nx)  # DTLZ3
#                 elif dtlz_type == 4:
#                     y = np.linspace(0.5 - 0.4, 0.5 + 0.4, nx)
#
#                 # y = np.linspace(0, 1, nx)
#                 xv, yv = np.meshgrid(x, y)
#                 x = np.vstack(xv.flatten())
#                 y = np.vstack(yv.flatten())
#                 X = np.hstack((x, y))
#                 n = x.shape[0]
#
#                 out = dict()
#                 G = np.zeros((n, problem.n_constr))
#                 problem._evaluate(X, out)
#                 F = out["F"]
#                 G = out["G"]
#             else:
#                 f1 = np.linspace(0, 10, nx)
#                 f2 = np.linspace(0, 10, nx)
#                 xv, yv = np.meshgrid(f1, f2)
#                 f1 = np.vstack(xv.flatten())
#                 f2 = np.vstack(yv.flatten())
#                 F = np.hstack((f1, f2))
#                 n = F.shape[0]
#
#             # plot_test_problem(F, problem, type=dtlz_type)
#
#             F1 = F[:, 0].flatten()
#             F2 = F[:, 1].flatten()
#
#             G[G <= 0] = 0
#             if G.ndim > 1:
#                 G = np.sum(G, axis=1)
#             index = G > 0
#
#             ref_dirs = UniformReferenceDirectionFactory(n_dim=problem.n_obj, n_points=21)._do()
#             PF = problem._calc_pareto_front(ref_dirs)
#
#             # plt.plot(F1, F2, 'o', color='#DCDCDC')  # '#A9A9A9'
#             # plt.plot(F1[index], F2[index], 'o', color='b')
#             # plt.plot(PF[:, 0], PF[:, 1], 'o', color='r')
#             ax[count].set_title(name)
#             ax[count].plot(F1, F2, 'o', color='#DCDCDC')  # '#A9A9A9'
#             ax[count].plot(F1[index], F2[index], 'o', color='b')
#             ax[count].plot(PF[:, 0], PF[:, 1], 'o', color='r')
#             ax[count].set_xlim([np.min(F1), np.max(F1)])
#             count += 1
#     plt.show()

        # print(F)

# class CDTLZ(DTLZ):
#
#     def __init__(self, dtlz, clist=None, rlist=None, **kwargs):
#
#         if clist is None or rlist is None:
#             raise Exception("Constraint number or parameter r cannot be None ")
#         if len(clist) != len(rlist):
#             raise Exception("Provide parameter r for all constrained problems")
#
#         self.rlist = anp.asarray(rlist)
#         self.clist = anp.asarray(clist)
#         self.dtlz = dtlz
#
#         super().__init__(n_var=dtlz.n_var, n_obj=dtlz.n_obj, **kwargs)
#         self.n_constr = len(anp.unique(clist))
#
#         if isinstance(dtlz, DTLZ1):
#             self.dtlz_type = 1
#         elif isinstance(dtlz, DTLZ2):
#             self.dtlz_type = 2
#         elif isinstance(dtlz, DTLZ3):
#             self.dtlz_type = 3
#         elif isinstance(dtlz, DTLZ4):
#             self.dtlz_type = 4
#         else:
#             raise Exception("DTLZ problem not supported.")
#
#     def _evaluate(self, X, out, *args, **kwargs):
#
#         self.dtlz._evaluate(X, out, *args, **kwargs)
#
#         g = []
#         for i in range(self.n_constr):
#             if self.clist[i] == 1:
#                 _g = constraint_c1(out["F"], self.rlist[i], self.dtlz_type)
#             elif self.clist[i] == 2:
#                 _g = constraint_c2(out["F"], self.rlist[i])
#             elif self.clist[i] == 3:
#                 _g = constraint_c3(out["F"], self.dtlz_type)
#             elif self.clist[i] == 4:
#                 _g = constraint_c4(out["F"], self.rlist[i])
#             else:
#                 _g = []
#
#             g.append(_g)
#
#         out["G"] = anp.column_stack(g)
#
#     def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
#
#         F = self.dtlz.pareto_front(ref_dirs, *args, **kwargs)
#
#         if anp.any(self.clist == 3):
#             a = anp.sqrt(anp.sum(F ** 2, 1) - 3 / 4 * anp.max(F ** 2, axis=1))
#             a = anp.expand_dims(a, axis=1)
#             a = anp.tile(a, [1, ref_dirs.shape[1]])
#             F = F / a
#         elif anp.any(self.clist == 2):
#             r = self.rlist[anp.where(self.clist == 2)][0]
#             _g = constraint_c2(F, r)
#             _g[_g <= 0] = 0
#             F = F[_g <= 0]
#         elif anp.any(self.clist == 4):
#             r = self.rlist[anp.where(self.clist == 4)][0]
#             _g = constraint_c4(F, r)
#             _g[_g <= 0] = 0
#             F = F[_g <= 0]
#
#         return F