from metamodels.siamese_net import SiameseMetamodel
from metamodels.nn_models import SiameseNet, ConstraintNet, ConstraintNet2
from metamodels.constraint_net import ConstraintMetamodel
from metamodels.regression_metamodel import GPRmodel, SVRmodel, KRRmodel


def get_model(framework_id=None, problem=None):
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
    else:  # if framework_id == '12B'
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

    return model_list