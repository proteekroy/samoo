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
from pymoo.model.evaluator import Evaluator
from frameworks.factory import Framework
import sys
import numpy as np
from pymoo.model.termination import Termination, get_termination
from pymoo.rand import random
from frameworks.framework_switching import FrameworkSwitching
from ensemble.candidate_select import framework_candidate_select

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class Samoo(GeneticAlgorithm):

    def __init__(self, ref_dirs,
                 framework_id=None,
                 metamodel_list=None,
                 acq_list=None,
                 framework_acq_dict=None,
                 aggregation=None,
                 disp=False,
                 lf_algorithm_list=None,
                 init_pop_size=None,
                 pop_size_per_epoch=None,
                 pop_size_per_algorithm=None,
                 pop_size_lf=None,
                 n_split=10,
                 n_gen_lf=100,
                 **kwargs):
        kwargs['individual'] = Individual(rank=np.inf, niche=-1, dist_to_niche=np.inf)
        set_if_none(kwargs, 'pop_size', init_pop_size)
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=1.0, eta_cross=15))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=None, eta_mut=20))
        set_if_none(kwargs, 'selection', TournamentSelection(func_comp=comp_by_cv_then_random))
        set_if_none(kwargs, 'survival', ReferenceDirectionSurvival(ref_dirs))
        set_if_none(kwargs, 'eliminate_duplicates', True)
        set_if_none(kwargs, 'disp', disp)
        super().__init__(**kwargs)

        self.func_display_attrs = disp_multi_objective
        self.init_pop_size = init_pop_size
        self.pop_size_lf = pop_size_lf
        self.pop_size_per_epoch = pop_size_per_epoch
        self.pop_size_per_algorithm = pop_size_per_algorithm
        self.framework_crossval = 10
        self.n_gen_lf = n_gen_lf
        self.ref_dirs = ref_dirs
        self.cur_ref_no = 0
        self.framework_id = framework_id
        self.metamodel_list = metamodel_list
        self.metamodel_list = self.metamodel_list
        self.acq_list = acq_list
        self.framework_acq_dict = framework_acq_dict
        self.aggregation = aggregation
        self.lf_algorithm_list = lf_algorithm_list
        self.n_split = n_split
        self.problem = None
        self.archive = None
        self.metamodel = None
        self.pop = None
        self.samoo_evaluator = SamooEvaluator()
        self.generative_algorithm = ['rga', 'rga_x', 'de']
        self.simultaneous_algorithm = ['mm_rga', 'nsga2', 'nsga3', 'moead']

    def _solve(self, problem, termination):
        if self.ref_dirs.shape[1] != problem.n_obj:
            raise Exception(
                "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                (self.ref_dirs.shape[1], problem.n_obj))
        return self.__solve(problem, termination)

    def __solve(self, problem, termination):
        self.n_gen = 1  # generation counter
        self.pop = self._initialize()  # initialize the first population and evaluate it
        self.pop_size = np.min([self.pop_size_lf, self.pop_size_per_epoch])
        self._init_samoo(problem, self.pop.get("X"))
        self.evaluator.n_eval = self.samoo_evaluator.n_eval
        self._each_iteration(self, first=True)
        self.samoo_evaluator.n_max_eval = termination.n_max_evals

        # while termination criterium not fulfilled
        while termination.do_continue(self):
            model_pop_X = self._next()  # do the next iteration
            self.pop = self._each_iteration_samoo(model_pop_X)  # callback for samoo
            self.n_gen += 1  # update generation counters
            self.evaluator.n_eval = self.samoo_evaluator.n_eval
            self._each_iteration(self)  # execute the callback function in the end of each generation

        self._finalize()

        return self.pop

    def _init_samoo(self, problem, X, **kwargs):
        self.problem = problem
        self.archive = dict()  # information about samoo saved here
        self.archive['x'] = np.empty([0, problem.n_var])
        self.archive['f'] = np.empty([0, problem.n_obj])
        self.archive['g'] = np.empty([0, np.max([problem.n_constr, 1])])

        self.framework = FrameworkSwitching(framework_id=self.framework_id,
                                            metamodel_list=self.metamodel_list,
                                            acq_list=self.acq_list,
                                            framework_acq_dict=self.framework_acq_dict,
                                            aggregation=self.aggregation,
                                            n_split=self.n_split,
                                            problem=problem,
                                            algorithm=self,
                                            ref_dirs=self.ref_dirs)

        self.samoo_problem = SamooProblem(problem=self.problem, framework=self.framework)  # problem wrapper
        self.cur_ref_no = 0
        self.framework.set_current_reference(self.cur_ref_no)
        self.archive = self.samoo_evaluator.eval(self.samoo_problem, X, archive=self.archive)
        self.func_eval = self.archive['x'].shape[0]

    def _each_iteration_samoo(self, X, **kwargs):
        self.archive = self.samoo_evaluator.eval(problem=self.samoo_problem, x=X, archive=self.archive)
        temp_pop = Population(0, individual=Individual())
        temp_pop = temp_pop.new("X", self.archive['x'], "F", self.archive['f'], "CV", self.archive['cv'], "G", self.archive['g'], "feasible", self.archive['feasible_index'])
        self.func_eval = self.archive['x'].shape[0]

        return temp_pop

    def _next(self):

        self.framework.train(x=self.archive["x"], f=self.archive["f"], g=self.archive["g"])
        out_pop = Population(0, individual=Individual())
        for fr in self.framework.best_frameworks:
            self.samoo_problem.framework = fr
            for lf_algorithm in self.lf_algorithm_list:

                if fr.type == 2:
                    if lf_algorithm in self.simultaneous_algorithm:

                        res = lf_minimize(problem=self.samoo_problem,
                                          method=lf_algorithm,
                                          method_args={'pop_size': self.pop_size_lf, 'ref_dirs': self.ref_dirs},
                                          termination=('n_gen', self.n_gen_lf),
                                          pf=self.pf,
                                          save_history=False,
                                          disp=False)

                        if self.pop_size_per_algorithm < len(res.pop):
                            res.pop = framework_candidate_select(fr.framework_id,
                                                                 ref_dirs=self.ref_dirs,
                                                                 pop=res.pop,
                                                                 n_select=self.pop_size_per_algorithm)
                        out_pop = out_pop.merge(res.pop)

                elif fr.type == 1:
                    if lf_algorithm in self.generative_algorithm:
                        # if fr.framework_id in ['11', '21']:
                        #    fr.train(x=self.archive["x"], f=self.archive["f"], g=self.archive["g"])
                        for i in range(len(self.ref_dirs)):
                            self.cur_ref_no = i
                            fr.set_current_reference(self.cur_ref_no)

                            if fr.framework_id in ['31', '41', '5']:
                                fr.train(x=self.archive["x"], f=self.archive["f"], g=self.archive["g"])

                            res = lf_minimize(problem=self.samoo_problem,
                                              method=lf_algorithm,
                                              method_args={'pop_size': self.pop_size_lf, 'ref_dirs': self.ref_dirs},
                                              termination=('n_gen', self.n_gen_lf),
                                              pf=self.pf,
                                              save_history=False,
                                              disp=False)

                            if np.any(res.pop.get("CV") <= 0):
                                I = res.pop.get("CV") <= 0
                                res.pop = res.pop[I.flatten()]
                                ind = res.pop[np.argmin(res.pop.get("F"))]
                            else:
                                ind = res.pop[np.argmin(res.pop.get("CV"))]

                            # # out_pop.append(ind)
                            # if len(out_pop) == 0:
                            #     out_pop = Population(1, individual=ind)
                            # else:
                            out_pop = out_pop.merge(Population(1, individual=ind))

        # if self.pop_size_per_epoch < len(out_pop):
        #     out_pop = self.candidate_select(ref_dirs=self.ref_dirs, pop=out_pop)

        return out_pop.get("X")

    def _next_simultaneous(self):

        self.framework.train(x=self.archive["x"], f=self.archive["f"], g=self.archive["g"])
        out_pop = []
        for lf_algorithm in self.lf_algorithm_list:
            res = lf_minimize(problem=self.samoo_problem,
                                   method=lf_algorithm,
                                   method_args={'pop_size': self.pop_size_lf, 'ref_dirs': self.ref_dirs},
                                   termination=('n_gen', self.n_gen_lf),
                                   pf=self.pf,
                                   save_history=False,
                                   disp=False)
            out_pop.append(res.pop)

        out_pop = np.row_stack(out_pop).view(Population)

        if self.pop_size_per_epoch < len(out_pop):
            out_pop = self.candidate_select(ref_dirs=self.ref_dirs, pop=out_pop)

        return out_pop.get("X")

    def _next_generative(self):

        if self.framework.framework_id in ['11', '21']:
            self.framework.train(x=self.archive["x"], f=self.archive["f"], g=self.archive["g"])

        out_pop_X = []
        for i in range(len(self.ref_dirs)):
            self.cur_ref_no = i
            self.framework.set_current_reference(self.cur_ref_no)

            if self.framework.framework_id in ['31', '41', '5']:
                self.framework.train(x=self.archive["x"], f=self.archive["f"], g=self.archive["g"])

            res = lf_minimize(problem=self.samoo_problem,
                                   method=self.lf_algorithm,
                                   method_args={'pop_size': self.pop_size_lf, 'ref_dirs': self.ref_dirs},
                                   termination=('n_gen', self.n_gen_lf),
                                   pf=self.pf,
                                   save_history=False,
                                   disp=False)

            if np.any(res.pop.get("CV") <= 0):
                I = res.pop.get("CV") <= 0
                res.pop = res.pop[I.flatten()]
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
        self.n_max_eval = np.inf

    def eval(self, problem=None, x=None, archive=None, **kwargs):
        if x.ndim == 1:
            n = 1
            x = np.expand_dims(x, 0)
        else:
            n = x.shape[0]
        f = np.zeros((n, problem.n_obj))
        g = np.zeros((n, np.max([problem.n_constr, 1])))

        if self.n_eval + x.shape[0] > self.n_max_eval:
            rest = self.n_max_eval - self.n_eval
            I = np.random.permutation(x.shape[0])
            I = I[:rest]
            x = x[I]
            f = f[I]
            g = g[I]

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
        archive['feasible_index'] = cv <= 0
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


def lf_get_alorithm(name):
    if name == 'ga':
        from pymoo.algorithms.so_genetic_algorithm import SingleObjectiveGeneticAlgorithm
        return SingleObjectiveGeneticAlgorithm
    elif name == 'nsga2':
        from pymoo.algorithms.nsga2 import NSGA2
        return NSGA2
    elif name == 'nsga3':
        from pymoo.algorithms.nsga3 import NSGA3
        return NSGA3
    elif name == 'unsga3':
        from pymoo.algorithms.unsga3 import UNSGA3
        return UNSGA3
    elif name == 'rnsga3':
        from pymoo.algorithms.rnsga3 import RNSGA3
        return RNSGA3
    elif name == 'moead':
        from pymoo.algorithms.moead import MOEAD
        return MOEAD
    elif name == 'de':
        from pymoo.algorithms.so_de import DifferentialEvolution
        return DifferentialEvolution
    elif name == 'rga_x':
        from algorithms.rga_x import RGATournamentSurvivalAlgorithm
        return RGATournamentSurvivalAlgorithm
    elif name == 'mm_rga':
        from algorithms.mm_rga import MMRGA
        return MMRGA
    else:
        raise Exception("Algorithm not known.")


def lf_minimize(problem,
             method,
             method_args={},
             termination=('n_gen', 200),
             **kwargs):
    """

    Minimization of function of one or more variables, objectives and constraints.

    This is used as a convenience function to execute several algorithms with default settings which turned
    out to work for a test problems. However, evolutionary computations utilizes the idea of customizing a
    meta-algorithm. Customizing the algorithm using the object oriented interface is recommended to improve the
    convergence.

    Parameters
    ----------

    problem : pymop.problem
        A problem object defined using the pymop framework. Either existing test problems or custom problems
        can be provided. please have a look at the documentation.
    method : string
        Algorithm that is used to solve the problem.
    method_args : dict
        Additional arguments to initialize the algorithm object
    termination : tuple
        The termination criterium that is used to stop the algorithm when the result is satisfying.

    Returns
    -------
    res : Result
        The optimization result represented as a ``Result`` object.

    """

    # create an evaluator defined by the termination criterium
    if not isinstance(termination, Termination):
        termination = get_termination(*termination, pf=kwargs.get('pf', None))

    # set a random random seed if not provided
    if 'seed' not in kwargs:
        kwargs['seed'] = random.randint(1, 10000)

    algorithm = lf_get_alorithm(method)(**method_args)
    res = algorithm.solve(problem, termination, **kwargs)

    return res

