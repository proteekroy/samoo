from optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.problems.g import *
from model_loader import get_model
import numpy as np
# create the optimization problem
# problem_name = 'tnk'
problem_name = 'osy'
lf_algorithm = 'mm_rga'  # ''rga_x'
framework_id = '6'
init_pop_size = 100
pop_size_per_epoch = 21  # 21
pop_size_lf = 100
method= 'simultaneous'  # 'generative'  # 'simultaneous' # 'generative'


# problem = get_problem(problem_name, n_var=10, n_obj=2)
problem = get_problem(problem_name)
# problem = get_problem(problem_name, n_var=10)
ref_dirs = UniformReferenceDirectionFactory(problem.n_obj, n_points=pop_size_per_epoch).do()
if lf_algorithm == 'nsga3' and problem_name.__contains__('dtlz'):
    pf = problem.pareto_front(ref_dirs=ref_dirs)
else:
    pf = np.loadtxt("../data/IGD/TNK.2D.pf")  # problem.pareto_front()

metamodel_list=dict()
metamodel_list['f'] = 'dacefit'  # 'gpr'

model_list = get_model(framework_id=framework_id,
                       problem=problem,
                       metamodel_list=metamodel_list,
                       uniform=True,
                       n_dir=pop_size_per_epoch)

res = minimize(problem=problem,
               method=method,  # 'generative',#'simultaneous',
               method_args={'framework_id': framework_id,
                            'ref_dirs': ref_dirs,
                            'model_list': model_list,
                            'lf_algorithm': lf_algorithm,
                            'init_pop_size': init_pop_size,
                            'pop_size_per_epoch': pop_size_per_epoch,
                            'pop_size_lf': pop_size_lf,
                            'n_gen_lf': 100,
                            },
               termination=('n_eval', 800),
               pf=pf,
               save_history=False,
               disp=True
               )
plotting.plot(pf, res.F, labels=["Pareto-front", "F"])
