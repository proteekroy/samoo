from pymoo.model.population import Population
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.model.individual import Individual
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.util.display import disp_multi_objective
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.nsga3 import ReferenceDirectionSurvival, comp_by_cv_then_random
from pymop.problems import *
from pymoo.optimize import minimize
from pymoo.model.evaluator import Evaluator
from frameworks.get_framework import get_framework
from frameworks.factory import Framework
import sys
import numpy as np
from abc import abstractmethod


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class Samoo(GeneticAlgorithm):

    def __init__(self, ref_dirs,
                 framework_id=12,
                 model_list=None,
                 disp=False,
                 lf_algorithm='nsga3',
                 init_pop_size=None,
                 pop_size_per_epoch=None,
                 pop_size_lf=None,
                 **kwargs):
        self.init_pop_size = init_pop_size
        self.pop_size_lf = pop_size_lf  # lf = low-fidelity
        self.pop_size_per_epoch = pop_size_per_epoch
        self.framework_crossval = 10
        self.n_gen_lf = 200
        self.framework_id = framework_id
        self.model_list = model_list
        self.ref_dirs = ref_dirs
        self.cur_ref_no = 0
        self.disp = disp
        self.lf_algorithm = lf_algorithm

        kwargs['individual'] = Individual(rank=np.inf, niche=-1, dist_to_niche=np.inf)
        set_if_none(kwargs, 'pop_size', self.init_pop_size)
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=1.0, eta_cross=30))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=None, eta_mut=20))
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_then_random))
        set_if_none(kwargs, 'survival', ReferenceDirectionSurvival(ref_dirs))
        set_if_none(kwargs, 'eliminate_duplicates', True)
        # set_if_none(kwargs, 'evaluator', SamooEvaluator())
        self.problem = None
        self.archive = None
        self.metamodel = None
        self.pop = None
        self.samoo_evaluator = SamooEvaluator()
        super().__init__(**kwargs)
        self.func_display_attrs = disp_multi_objective

    def _solve(self, problem, termination):
        if self.ref_dirs.shape[1] != problem.n_obj:
            raise Exception(
                "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                (self.ref_dirs.shape[1], problem.n_obj))

        return self.__solve(problem, termination)

    def __solve(self, problem, termination):

        # generation counter
        self.n_gen = 1
        # initialize the first population and evaluate it
        self.pop = self._initialize()
        self.pop_size = np.min([self.pop_size_lf, self.pop_size_per_epoch])
        self._init_samoo(problem, self.pop.get("X"))
        # super().pop = self.pop
        self.evaluator.n_eval = self.samoo_evaluator.n_eval
        self._each_iteration(self, first=True)

        # while termination criterium not fulfilled
        while termination.do_continue(self):
            # do the next iteration
            model_pop_X = self._next()
            # callback for samoo
            self.pop = self._each_iteration_samoo(model_pop_X)
            # update counters
            self.n_gen += 1
            self.evaluator.n_eval = self.samoo_evaluator.n_eval
            # execute the callback function in the end of each generation
            self._each_iteration(self)

        self._finalize()

        return self.pop

    def _init_samoo(self, problem, X, **kwargs):
        self.problem = problem
        self.archive = dict()  # information about samoo saved here
        self.archive['x'] = np.empty([0, problem.n_var])
        self.archive['f'] = np.empty([0, problem.n_obj])
        self.archive['g'] = np.empty([0, np.max([problem.n_constr, 1])])
        self.framework = get_framework(framework_id=self.framework_id,
                                       framework_crossval=self.framework_crossval,
                                       problem=problem,
                                       algorithm=self,
                                       ref_dirs=self.ref_dirs,
                                       curr_ref_id=self.cur_ref_no,
                                       model_list=self.model_list)  # create frameworks
        self.samoo_problem = SamooProblem(problem=self.problem, framework=self.framework)  # problem wrapper
        self.cur_ref_no = 0
        self.framework.set_current_reference(self.cur_ref_no)
        self.archive = self.samoo_evaluator.eval(self.samoo_problem, X, archive=self.archive)
        self.func_eval = self.archive['x'].shape[0]

    def _each_iteration_samoo(self, X, **kwargs):

        # if not isinstance(pop, Individual):
        #     if self.pop_size_per_epoch > len(pop):
        #         pop = self.candidate_select(ref_dirs=self.ref_dirs, pop=pop)
        #
        # if isinstance(pop, Individual):
        #     X = pop.X
        # elif isinstance(pop, Population):
        # X = pop.get("X")
        # else:
        #     X = pop

        self.archive = self.samoo_evaluator.eval(problem=self.samoo_problem, x=X, archive=self.archive)
        temp_pop = Population(0, individual=Individual())
        temp_pop = temp_pop.new("X", self.archive['x'], "F", self.archive['f'], "CV", self.archive['cv'], "G", self.archive['g'], "feasible", self.archive['feasible_index'])
        self.func_eval = self.archive['x'].shape[0]

        return temp_pop

    def _next(self):

        self.framework.train(x=self.archive["x"], f=self.archive["f"], g=self.archive["g"])
        res = minimize(problem=self.samoo_problem,
                               method=self.lf_algorithm,
                               method_args={'pop_size': self.pop_size_lf, 'ref_dirs': self.ref_dirs},
                               termination=('n_gen', self.n_gen_lf),
                               pf=self.pf,
                               save_history=False,
                               disp=False)
        return res.pop.get("X")

    def candidate_select(self, ref_dirs=None, pop=None):

        if self.pop_size_per_epoch == 1:
            F = pop.get("F")
            if F.shape[1] > 1:  # if it is multi-objective, pick the middle one
                ref = (1/F.shape[1])*np.ones(F.shape[1])
                _F = np.sum(ref*F, 1)
                pop = pop[np.argmin(_F)]
            else:  # single-objective
                if np.any(pop.get("CV") <= 0):
                    pop = pop[np.argmin(pop.get("F"))]
                else:
                    pop = pop[np.argmin(pop.get("CV"))]
            return pop

        a_out = dict()
        out_pop = []
        for i in range(len(ref_dirs)):

            Framework.prepare_aggregate_data(f=pop.get("F"),
                                             g=pop.get("G"),
                                             out=a_out,
                                             f_aggregate='asf',
                                             g_aggregate=None,
                                             m5_fg_aggregate='asfcv',
                                             ref_dirs=ref_dirs,
                                             curr_ref_id=i)
            index = np.argsort(a_out['S5'])
            j = 0
            while pop[index[j]] in out_pop:
                j += 1
            out_pop.append(pop[index[j]])

        return np.column_stack(out_pop)


class Simultaneous(Samoo):
    def __init__(self, ref_dirs,
                 framework_id=12,
                 model_list=None,
                 disp=False,
                 lf_algorithm='nsga3',
                 init_pop_size=100,
                 pop_size_per_epoch=100,
                 pop_size_lf=100,
                 **kwargs):
        super().__init__(ref_dirs, framework_id, model_list, disp, lf_algorithm, init_pop_size,
                                         pop_size_per_epoch, pop_size_lf, **kwargs)


class Generative(Samoo):

    def __init__(self, ref_dirs,
                 framework_id=12,
                 model_list=None,
                 disp=False,
                 lf_algorithm='nsga3',
                 init_pop_size=100,
                 pop_size_per_epoch=100,
                 pop_size_lf=100,
                 **kwargs):
        super().__init__(ref_dirs, framework_id, model_list, disp, lf_algorithm, init_pop_size,
                                         pop_size_per_epoch, pop_size_lf, **kwargs)

    def _next(self):

        if self.framework.framework_id in  ['11', '21']:
            self.framework.train(x=self.archive["x"], f=self.archive["f"], g=self.archive["g"])

        out_pop_X = []
        for i in range(len(self.ref_dirs)):
            self.cur_ref_no = i
            self.framework.set_current_reference(self.cur_ref_no)

            if self.framework.framework_id in ['31', '41']:
                self.framework.train(x=self.archive["x"], f=self.archive["f"], g=self.archive["g"])

            res = minimize(problem=self.samoo_problem,
                                   method=self.lf_algorithm,
                                   method_args={'pop_size': self.pop_size_lf, 'ref_dirs': self.ref_dirs},
                                   termination=('n_gen', self.n_gen_lf),
                                   pf=self.pf,
                                   save_history=False,
                                   disp=False)

            if np.any(res.pop.get("CV") <= 0):
                ind = res.pop[np.argmin(res.pop.get("F"))]
            else:
                ind = res.pop[np.argmin(res.pop.get("CV"))]

            out_pop_X.append(ind.X)

        out_pop_X = np.row_stack(out_pop_X)

        return out_pop_X


# wrapper for pymop problem which is used by metamodel based optimization
class SamooProblem(Problem):

    def __init__(self, problem, framework, *args, **kwargs):
        self.problem = problem
        self.framework = framework
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj, n_constr=problem.n_constr, type_var=problem.type_var,
                         xl=problem.xl, xu=problem.xu)

    def _evaluate_high_fidelity(self, x, f, g, *args, **kwargs):
        out = dict()
        self.problem._evaluate(x, out, *args, **kwargs)
        f[:, :] = out["F"]
        if self.problem.n_constr > 0:
            g[:, :] = out["G"]

    def _evaluate(self, x, out, *args, **kwargs):
        samoo_output = dict()
        self.framework.predict(x, samoo_output, *args, **kwargs)
        out["F"] = samoo_output['F']
        out["G"] = samoo_output['G']


class SamooEvaluator(Evaluator):

    def __init__(self):
        super(SamooEvaluator, self).__init__()
        self.n_eval = 0

    def eval(self, problem=None, x=None, archive=None, **kwargs):
        if x.ndim == 1:
            n = 1
            x = np.expand_dims(x, 0)
        else:
            n = x.shape[0]
        f = np.zeros((n, problem.n_obj))
        g = np.zeros((n, np.max([problem.n_constr, 1])))
        problem._evaluate_high_fidelity(x=x, f=f, g=g)

        archive['x'] = np.concatenate((x, archive['x']), axis=0)
        archive['f'] = np.concatenate((f, archive['f']), axis=0)
        archive['g'] = np.concatenate((g, archive['g']), axis=0)
        cv = np.copy(archive['g'])
        index = np.any(archive['g'] > 0, axis=1)
        cv[archive['g'] <= 0] = 0
        cv = np.sum(cv, axis=1)
        acv = np.sum(archive['g'], axis=1)
        acv[index] = np.copy(cv[index])
        archive['feasible_index'] = cv == 0
        archive['feasible_index'] = np.vstack(np.asarray(archive['feasible_index']).flatten())
        archive['cv'] = np.vstack(np.asarray(cv).flatten())
        archive['acv'] = np.vstack(np.asarray(acv).flatten())

        feasible = archive['f'][archive['feasible_index'][:, 0], :]
        if feasible.size > 0:
            nd = NonDominatedSorting()
            index = nd.do(F=feasible, only_non_dominated_front=True)
            archive['non_dominated_front'] = archive['x'][index]
        else:
            archive['non_dominated_front'] = np.empty([0, f.size])

        self.n_eval = archive["x"].shape[0]
        return archive
