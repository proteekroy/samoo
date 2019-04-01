from optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from model_loader import get_acq_function
import numpy as np
from collections import defaultdict

lf_algorithm_list = ['nsga2']
framework_id = ['11', '12'] #, '12', '21', '22', '31', '5', '6A', '6B']  # ['6A', '6B']
aggregation = defaultdict(list)
aggregation['l'].append('asf')
aggregation['G'].append('cv')
aggregation['G'].append('acv')
aggregation['fg_M5'].append('asfcv')
aggregation['fg_M6'].append('minasfcv')
metamodel_list = ['dacefit']

init_pop_size = 100
pop_size_per_epoch = 21
pop_size_lf = 100

problem_name = 'tnk'
# problem = get_problem(problem_name, n_var=10, n_obj=2)
problem = get_problem(problem_name)
# problem = get_problem(problem_name, n_var=10)
ref_dirs = UniformReferenceDirectionFactory(problem.n_obj, n_points=pop_size_per_epoch).do()
if problem_name.__contains__('dtlz'):
    pf = problem.pareto_front(ref_dirs=ref_dirs)
else:
    pf = np.loadtxt("../data/IGD/TNK.2D.pf")  # problem.pareto_front()


acq_list, framework_acq_dict = get_acq_function(framework_id=framework_id,
                                                aggregation=aggregation,
                                                problem=problem,
                                                n_dir=pop_size_per_epoch)

res = minimize(problem=problem,
               method='samoo',
               method_args={# 'seed': 53,
                            'method': 'samoo',
                            'framework_id': framework_id,
                            'ref_dirs': ref_dirs,
                            'metamodel_list': metamodel_list,
                            'acq_list': acq_list,
                            'framework_acq_dict': framework_acq_dict,
                            'aggregation': aggregation,
                            'lf_algorithm_list': lf_algorithm_list,
                            'init_pop_size': init_pop_size,
                            'pop_size_per_epoch': pop_size_per_epoch,
                            'pop_size_lf': pop_size_lf,
                            'n_gen_lf': 100,
                            'n_split': 2
                            },
               termination=('n_eval', 300),
               pf=pf,
               save_history=False,
               disp=True
               )
plotting.plot(pf, res.F, labels=["Pareto-front", "F"])
