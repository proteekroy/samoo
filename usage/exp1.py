from optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
# from pymop.problems.g import *
from model_loader import get_model

# create the optimization problem
# problem_name = 'tnk'
problem_name = 'zdt2'
lf_algorithm = 'nsga3'
framework_id = '12A'
init_pop_size = 100
pop_size_per_epoch = 100
pop_size_lf = 100


# problem = get_problem(problem_name, n_var=10, n_obj=2)
# problem = get_problem(problem_name)
problem = get_problem(problem_name, n_var=10)
ref_dirs = UniformReferenceDirectionFactory(problem.n_obj, n_points=21).do()
if lf_algorithm == 'nsga3' and problem_name.__contains__('dtlz'):
    pf = problem.pareto_front(ref_dirs=ref_dirs)
else:
    pf = problem.pareto_front()


model_list = get_model(framework_id=framework_id, problem=problem)
res = minimize(problem=problem,
               method='samoo',
               method_args={'framework_id': framework_id,
                            'framework_crossval': 10,
                            'ref_dirs': ref_dirs,
                            'model_list': model_list,
                            'disp': False,
                            'lf_algorithm': lf_algorithm,
                            'init_pop_size': init_pop_size,
                            'pop_size_per_epoch': pop_size_per_epoch,
                            'pop_size_lf': pop_size_lf
                            },
               termination=('n_eval', 1000),
               pf=pf,
               save_history=False,
               disp=True
               )
plotting.plot(pf, res.F, labels=["Pareto-front", "F"])
