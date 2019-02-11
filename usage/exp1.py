from optimize import minimize
from pymoo.util import plotting
from pymop.factory import get_problem
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from metamodels.siamese_net import SiameseMetamodel
from metamodels.nn_models import SiameseNet, ConstraintNet, ConstraintNet2
from metamodels.constraint_net import ConstraintMetamodel
from metamodels.regression_metamodel import GPRmodel, SVRmodel, KRRmodel
from pymop.problems.g import *

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


# #################------DEFINE METAMODEL-----############################## #

model_list = dict()

if framework_id == 'hybrid' or framework_id == '12A':
    if framework_id == '12A':
        model_list['f'] = GPRmodel()  # SVRmodel()  # GPRmodel()
    else:
        f_net = SiameseNet(n_var=problem.n_var, n_obj=problem.n_obj, hidden_layer_length=20,
                           embed_length=10)
        model_list['f'] = SiameseMetamodel(problem.n_var, problem.n_obj, neuralnet=f_net,
                                           total_epoch=300, disp=False,
                                           best_accuracy_model=False,
                                           batch_size=10)
    model_list['gpr'] = SVRmodel()  # GPRmodel()
else: # if framework_id == '12B'
    f_net = SiameseNet(n_var=problem.n_var, n_obj=problem.n_obj, hidden_layer_length=7,
                       embed_length=3)
    model_list['f'] = SiameseMetamodel(problem.n_var, problem.n_obj, neuralnet=f_net,
                                       total_epoch=100, disp=False,
                                       best_accuracy_model=False,
                                       batch_size=10)
    if problem.n_constr > 0:
        g_net = ConstraintNet(n_var=problem.n_var, n_constr=problem.n_constr)
        # g_net = SiameseNet(n_var=problem.n_var, n_obj=problem.n_constr)
        model_list['g'] = ConstraintMetamodel(problem.n_var, problem.n_constr,
                                              neuralnet=g_net, total_epoch=300,
                                              disp=False, best_accuracy_model=False)
        # model_list['g'] = SiameseMetamodel(problem.n_var, problem.n_constr, neuralnet=g_net, total_epoch=200)

# #################------RUN SAMOO-----############################## #

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
