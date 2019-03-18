from metamodels.siamese_net import SiameseMetamodel
from metamodels.nn_models import SiameseNet, ConstraintNet, ConstraintNet2
from metamodels.constraint_net import ConstraintMetamodel
from metamodels.regression_metamodel import GPRmodel, SVRmodel, KRRmodel


def get_model(framework_id=None, problem=None, metamodel_list=None, uniform=True, n_dir=-1):

    if len(metamodel_list) == 0:
        raise Exception('Metamodel not provided.')

    model_list = dict()
    if uniform:
        metamodel_name = metamodel_list['f']
        if framework_id.lower() in ['11', '12', '21', '22']:
            for i in range(problem.n_obj):
                model_list["f" + str(i + 1)] = return_metamodel_object(metamodel_name, problem)

        if framework_id.lower() in ['11', '12', '31', '32']:
            for j in range(problem.n_constr):
                model_list["g" + str(j + 1)] = return_metamodel_object(metamodel_name, problem)

        if framework_id.lower() in ['21', '22', '41', '42']:

            model_list["G"] = return_metamodel_object(metamodel_name, problem)

        if framework_id.lower() in ['31', '32', '41', '42']:

            for i in range(n_dir):
                model_list["f" + str(i + 1)] = return_metamodel_object(metamodel_name, problem)

    else:
        raise Exception('Different metamodels for different functions is not supported.')

    return model_list


def return_metamodel_object(metamodel_name, problem):
    if metamodel_name == 'gpr':
        return GPRmodel()
    elif metamodel_name == 'krr':
        return KRRmodel()
    elif metamodel_name == 'svr':
        return SVRmodel()
    elif metamodel_name == 'siamese':
        net = SiameseNet(n_var=problem.n_var, n_obj=problem.n_obj, hidden_layer_length=20, embed_length=10)
        return SiameseMetamodel(problem.n_var, problem.n_obj, neuralnet=net,
                               total_epoch=300, disp=False,
                               best_accuracy_model=False,
                               batch_size=10)
    elif metamodel_name == 'constrnet':
        net = ConstraintNet(n_var=problem.n_var, n_constr=problem.n_constr)
        return ConstraintMetamodel(problem.n_var, problem.n_constr,
                                   neuralnet=net, total_epoch=300,
                                   disp=False, best_accuracy_model=False)
