import numpy as np
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.default_operators import set_if_none
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection, compare
from pymoo.util.display import disp_single_objective


class RGATournamentSurvivalAlgorithm(GeneticAlgorithm):

    def __init__(self, **kwargs):
        self.tournament_selection = TournamentSelection(func_comp=comp_by_cv_and_fitness)
        set_if_none(kwargs, 'pop_size', 100)
        set_if_none(kwargs, 'sampling', RandomSampling())
        set_if_none(kwargs, 'selection', self.tournament_selection)
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob_cross=0.95, eta_cross=15))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob_mut=None, eta_mut=20))
        set_if_none(kwargs, 'survival', RGA_XSurvival(self.tournament_selection))
        set_if_none(kwargs, 'eliminate_duplicates', True)

        super().__init__(**kwargs)
        self.func_display_attrs = disp_single_objective


class RGA_XSurvival(Survival):

    def __init__(self, selection) -> None:
        super().__init__(True)
        self.tournament_selection = selection

    def _do(self, pop, n_survive, out=None, **kwargs):

        if pop.get("F").shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective problems!")

        index = self.tournament_selection._do(pop, n_survive).flatten()
        return pop[index[:n_survive]]


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
