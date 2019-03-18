from pymoo.model.termination import MaximumFunctionCallTermination, MaximumGenerationTermination, IGDTermination, \
    Termination, get_termination
from pymoo.rand import random
from samoo import *


def get_alorithm(name):
    if name == 'simultaneous':
        return Simultaneous
    elif name == 'generative':
        return Generative
    else:
        raise Exception("Algorithm not known.")

def minimize(problem,
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
        A problem object defined using the pymop frameworks. Either existing test problems or custom problems
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

    algorithm = get_alorithm(method)(**method_args)
    res = algorithm.solve(problem, termination, **kwargs)

    return res
