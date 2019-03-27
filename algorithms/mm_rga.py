import numpy as np
from pymoo.model.population import Population
from pymoo.model.individual import Individual
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare
from pymoo.util.display import disp_single_objective
from algorithms.rga_x import RGA_XSurvival
from pymoo.model.survival import split_by_feasibility


class MMRGA(GeneticAlgorithm):

    def __init__(self, **kwargs):
        self.tournament_selection = TournamentSelection(func_comp=comp_by_cv_and_fitness)
        set_if_none(kwargs, 'pop_size', 100)
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'selection', MMRGATournamentSelection(self.tournament_selection))
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=0.95, eta_cross=15))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=None, eta_mut=20))
        set_if_none(kwargs, 'survival', MMRGASurvivalSelection(self.tournament_selection))
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)
        self.func_display_attrs = disp_single_objective


class MMRGATournamentSelection(TournamentSelection):

    def __init__(self, selection):
        super().__init__(True)
        self.tournament_selection = selection

    def _do(self, pop, n_select, n_parents=1, **kwargs):

        f = pop.get("F")
        # cv = pop.get("CV")
        # f = mm_rga_fitness_assignment(f, cv)
        S = np.min(f, axis=1)

        # temp_pop = Population(0, individual=Individual())
        # temp_pop = temp_pop.new("X", pop.get("X"), "F", S,
        #                         "CV", pop.get("CV"), "G", pop.get("G"),
        #                         "feasible", pop.get("feasible"))

        pop.set("F", S)
        index = self.tournament_selection._do(pop, n_select, n_parents=n_parents, **kwargs)
        pop.set("F", f)
        return index


class MMRGASurvivalSelection(Survival):

    def __init__(self, selection) -> None:
        super().__init__(True)
        self.tournament_selection = selection

    def do(self, problem, pop, n_survive, **kwargs):

        # if the split should be done beforehand
        survivors = self._do(pop, n_survive, **kwargs)

        return survivors

    def _do(self, pop, n_survive, out=None, **kwargs):

        if len(pop) == n_survive:
            return pop

        cv = pop.get("CV")
        f = pop.get("F")
        f = mm_rga_fitness_assignment(f, cv)

        survivors = []
        for i in range(f.shape[1]):
            index = np.argsort(f[:, i])
            j = 0
            while index[j] in survivors:
                j += 1
            survivors.append(index[j])

        return pop[survivors]


def mm_rga_fitness_assignment(f, cv):

    feasible_index = (cv <= 0).flatten()  # find feasible index
    infeasible_index = (cv > 0).flatten()
    size = np.sum(infeasible_index)
    if size > 1:
        worst_of_all_ref_dir = np.max(f[feasible_index])
        val_infeasible_sol = cv[infeasible_index] + worst_of_all_ref_dir
        f[infeasible_index, :] = np.tile(val_infeasible_sol, (1, f.shape[1]))

    return f

def comp_by_cv_and_fitness(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = compare(a, pop[a].F, b, pop[b].F, method='smaller_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int)
