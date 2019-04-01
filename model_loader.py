from collections import defaultdict


# find all possible acquisition functions that is needed to run the frameworks
def get_acq_function(framework_id=None,
                     aggregation=None,
                     problem=None,
                     n_dir=-1):
    acq_list = []
    framework_acq_dict = defaultdict(list)
    if len(list(set(['11', '12', '21', '22']).intersection(framework_id))) > 0:
        for i in range(problem.n_obj):
            string = "f" + str(i + 1)
            acq_list.append(string)
            framework_acq_dict['11'].append(string)
            framework_acq_dict['12'].append(string)
            framework_acq_dict['21'].append(string)
            framework_acq_dict['22'].append(string)

    if len(list(set(['11', '12', '31', '32']).intersection(framework_id))) > 0:
        for j in range(problem.n_constr):
            string = "g" + str(j + 1)
            acq_list.append(string)
            framework_acq_dict['11'].append(string)
            framework_acq_dict['12'].append(string)
            framework_acq_dict['31'].append(string)
            framework_acq_dict['32'].append(string)

    if len(list(set(['21', '22', '41', '42']).intersection(framework_id))) > 0:
        for i in aggregation['G']:
            string = "G_" + str(i)
            acq_list.append(string)
            framework_acq_dict['21'].append(string)
            framework_acq_dict['22'].append(string)
            framework_acq_dict['41'].append(string)
            framework_acq_dict['42'].append(string)

    if len(list(set(['31', '32', '41', '42']).intersection(framework_id))) > 0:
        for i in range(n_dir):
            for j in aggregation['l']:
                string = "l" + str(i + 1) + '_'+str(j)
                acq_list.append(string)
                framework_acq_dict['31'].append(string)
                framework_acq_dict['32'].append(string)
                framework_acq_dict['41'].append(string)
                framework_acq_dict['42'].append(string)

    if len(list(set(['5']).intersection(framework_id))) > 0:
        for i in range(n_dir):
            for j in aggregation['fg_M5']:
                string = "fg_M5_" + str(i + 1) + '_'+str(j)
                acq_list.append(string)
                framework_acq_dict['5'].append(string)

    if len(list(set(['6B', '6A']).intersection(framework_id))) > 0:
        for j in aggregation['fg_M6']:
            if j not in ['asfcv']:
                string = "fg_M6_0_"+str(j)
                acq_list.append(string)
                framework_acq_dict['6B'].append(string)
            else:
                for i in range(n_dir):
                    string = "fg_M6_" + str(i + 1) + '_' + str(j)
                    acq_list.append(string)
                    framework_acq_dict['6A'].append(string)

    return acq_list, framework_acq_dict
