from frameworks.factory import Framework
import numpy as np
from frameworks.get_framework import get_framework
from metamodels.model_selection import select_best_metamodel, \
    prepare_data, return_metamodel_object
from scipy.stats import ranksums


class FrameworkSwitching(Framework):
    def __init__(self,
                 framework_id=None,
                 metamodel_list=None,
                 acq_list=None,
                 framework_acq_dict=None,
                 aggregation=None,
                 n_split=None,
                 problem=None,
                 algorithm=None,
                 ref_dirs=None,
                 *args,
                 **kwargs
                 ):
        super().__init__(framework_id=framework_id,
                         metamodel_list=metamodel_list,
                         acq_list=acq_list,
                         framework_acq_dict=framework_acq_dict,
                         aggregation=aggregation,
                         n_split=n_split,
                         problem=problem,
                         algorithm=algorithm,
                         ref_dirs=ref_dirs,
                         *args,
                         **kwargs)

        self.best_frameworks = []  # list of best frameworks
        self.model_list = {}
        self.error_d = None
        self.actual_data = None
        self.prediction_data = None
        self.best_frameworks_acq_list = None

    def select_framework(self, x, f, g, *args, **kwargs):

        flag = False
        if len(self.framework_id) == 1 \
                and len(self.aggregation['l']) <= 1 \
                and len(self.aggregation['G']) <= 1 \
                and len(self.aggregation['fg_M5']) <=1 \
                and len(self.aggregation['fg_M6']) <=1\
                and len(self.metamodel_list) <= 1:

                fr = get_framework(framework_id=self.framework_id[0],
                                   problem=self.problem,
                                   algorithm=self.algorithm,
                                   model_list=self.model_list,
                                   ref_dirs=self.ref_dirs,
                                   f_aggregate_func=''.join(self.aggregation['l']),
                                   g_aggregate_func=''.join(self.aggregation['G']),
                                   m5_fg_aggregate_func=''.join(self.aggregation['fg_M5']),
                                   m6_fg_aggregate_func=''.join(self.aggregation['fg_M6']),
                                   curr_ref_id=self.curr_ref_id)
                self.best_frameworks = []
                self.best_frameworks.append(fr)
                self.best_frameworks_acq_list = []
                # find out all acquisition functions for the best frameworks
                for val in self.framework_acq_dict[fr.framework_id]:
                    if val not in self.best_frameworks_acq_list:
                        self.best_frameworks_acq_list.append(val)
                        self.model_list[val] = return_metamodel_object(metamodel_name=self.metamodel_list[0], problem=self.problem)
                flag = True

        if not flag:
            # select best metamodels for each acquisition functions
            self.model_list, self.error_d, self.actual_data, self.prediction_data = select_best_metamodel(
                                                                         x=x,
                                                                         f=f,
                                                                         g=g,
                                                                         acq_func=self.acq_list,
                                                                         ref_dirs=self.ref_dirs,
                                                                         metamodel_list=self.metamodel_list,
                                                                         problem=self.problem,
                                                                         n_split=self.n_split)

            framework_list = []  # object of frameworks
            lowest_median = np.inf
            lowest_sep = np.zeros([self.n_split])
            lowest_median_fr = self.framework_id[0]

            # Make all combination of framework type and aggregation function and check
            # their selection error probability
            for fr_id in self.framework_id:

                if fr_id in ['11']:
                    for f_aggregation_func in self.aggregation['l']:
                        fr = get_framework(framework_id=fr_id,
                                           problem=self.problem,
                                           algorithm=self.algorithm,
                                           model_list=self.model_list,
                                           ref_dirs=self.ref_dirs,
                                           f_aggregate_func=f_aggregation_func,
                                           curr_ref_id=self.curr_ref_id)
                        sep = fr.calculate_sep(self.problem, self.actual_data, self.prediction_data, self.n_split)
                        fr.sep = sep
                        framework_list.append(fr)
                        if np.median(np.asarray(sep)) < lowest_median:
                            lowest_median = np.median(np.asarray(sep))
                            lowest_median_fr = fr

                elif fr_id in ['12']:
                    fr = get_framework(framework_id=fr_id,
                                       problem=self.problem,
                                       algorithm=self.algorithm,
                                       model_list=self.model_list,
                                       ref_dirs=self.ref_dirs,
                                       curr_ref_id=self.curr_ref_id)
                    sep = fr.calculate_sep(self.problem, self.actual_data, self.prediction_data, self.n_split)
                    fr.sep = sep
                    framework_list.append(fr)
                    if np.median(np.asarray(sep)) < lowest_median:
                        lowest_median = np.median(np.asarray(sep))
                        lowest_median_fr = fr

                elif fr_id in ['21', '22']:
                    for g_aggregation_func in self.aggregation['G']:
                        fr = get_framework(framework_id=fr_id,
                                           problem=self.problem,
                                           algorithm=self.algorithm,
                                           model_list=self.model_list,
                                           ref_dirs=self.ref_dirs,
                                           g_aggregate_func=g_aggregation_func,
                                           curr_ref_id=self.curr_ref_id)
                        sep = fr.calculate_sep(self.problem, self.actual_data, self.prediction_data, self.n_split)
                        fr.sep = sep
                        framework_list.append(fr)
                        if np.median(np.asarray(sep)) < lowest_median:
                            lowest_median = np.median(np.asarray(sep))
                            lowest_median_fr = fr
                            lowest_sep = sep

                elif fr_id in ['31', '32']:

                    for f_aggregation_func in self.aggregation['l']:
                        fr = get_framework(framework_id=fr_id,
                                           problem=self.problem,
                                           algorithm=self.algorithm,
                                           model_list=self.model_list,
                                           ref_dirs=self.ref_dirs,
                                           f_aggregate_func=f_aggregation_func,
                                           curr_ref_id=self.curr_ref_id)
                        sep = fr.calculate_sep(self.problem, self.actual_data, self.prediction_data, self.n_split)
                        fr.sep = sep
                        framework_list.append(fr)
                        if np.median(np.asarray(sep)) < lowest_median:
                            lowest_median = np.median(np.asarray(sep))
                            lowest_median_fr = fr
                            lowest_sep = sep

                elif fr_id in ['41', '42']:

                    for f_aggregation_func in self.aggregation['l']:
                        for g_aggregation_func in self.aggregation['G']:
                            fr = get_framework(framework_id=fr_id,
                                               problem=self.problem,
                                               algorithm=self.algorithm,
                                               model_list=self.model_list,
                                               ref_dirs=self.ref_dirs,
                                               f_aggregate_func=f_aggregation_func,
                                               g_aggregate_func=g_aggregation_func,
                                               curr_ref_id=self.curr_ref_id)
                            sep = fr.calculate_sep(self.problem, self.actual_data, self.prediction_data, self.n_split)
                            fr.sep = sep
                            framework_list.append(fr)
                            if np.median(np.asarray(sep)) < lowest_median:
                                lowest_median = np.median(np.asarray(sep))
                                lowest_median_fr = fr
                                lowest_sep = sep

                elif fr_id in ['5']:
                    for fg_aggregation_func in self.aggregation['fg_M5']:
                        fr = get_framework(framework_id=fr_id,
                                           problem=self.problem,
                                           algorithm=self.algorithm,
                                           model_list=self.model_list,
                                           ref_dirs=self.ref_dirs,
                                           m5_fg_aggregate_func=fg_aggregation_func,
                                           curr_ref_id=self.curr_ref_id)
                        sep = fr.calculate_sep(self.problem, self.actual_data, self.prediction_data, self.n_split)
                        fr.sep = sep
                        framework_list.append(fr)
                        if np.median(np.asarray(sep)) < lowest_median:
                            lowest_median = np.median(np.asarray(sep))
                            lowest_median_fr = fr
                            lowest_sep = sep

                elif fr_id in ['6A', '6B']:
                    for fg_aggregation_func in self.aggregation['fg_M6']:
                        fr = get_framework(framework_id=fr_id,
                                           problem=self.problem,
                                           algorithm=self.algorithm,
                                           model_list=self.model_list,
                                           ref_dirs=self.ref_dirs,
                                           m6_fg_aggregate_func=fg_aggregation_func,
                                           curr_ref_id=self.curr_ref_id)
                        sep = fr.calculate_sep(self.problem, self.actual_data, self.prediction_data, self.n_split)
                        fr.sep = sep
                        framework_list.append(fr)
                        if np.median(np.asarray(sep)) < lowest_median:
                            lowest_median = np.median(np.asarray(sep))
                            lowest_median_fr = fr
                            lowest_sep = sep

            self.best_frameworks_acq_list = []
            self.best_frameworks = []
            for fr in framework_list:
                if fr is lowest_median_fr:
                    self.best_frameworks.append(fr)
                    # find out all acquisition functions for the best frameworks
                    for val in self.framework_acq_dict[fr.framework_id]:
                        if val not in self.best_frameworks_acq_list:
                            self.best_frameworks_acq_list.append(val)

                else:
                    val = fr.sep
                    # perform wilcoxon ranksum test to find out equivalent frameworks
                    p_value = ranksums(lowest_sep, val)[1]  # p-value of wilcoxon ranksum test
                    if p_value > 0.05:
                        self.best_frameworks.append(fr)
                        for val in self.framework_acq_dict[fr.framework_id]:
                            if val not in self.best_frameworks_acq_list:
                                self.best_frameworks_acq_list.append(val)

        return

    def train(self, x, f, g, *args, **kwargs):

        self.select_framework(x, f, g, *args, **kwargs)

        for acq in self.best_frameworks_acq_list:
            d = prepare_data(x=x, f=f, g=g, acq_func=[acq], ref_dirs=self.ref_dirs,
                             curr_ref_id=-1)
            self.model_list[acq].train(x, d[acq])

    def predict(self, x, out, *args, **kwargs):

        for acq in self.best_frameworks_acq_list:
            out[acq] = self.model_list[acq].predict(x)

        return out
