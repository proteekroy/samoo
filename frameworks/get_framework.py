from frameworks.framework11 import Framework11
from frameworks.framework12 import Framework12
from frameworks.framework21 import Framework21
from frameworks.framework22 import Framework22
from frameworks.framework31 import Framework31
from frameworks.framework32 import Framework32
from frameworks.framework41 import Framework41
from frameworks.framework42 import Framework42
from frameworks.framework5 import Framework5
from frameworks.framework6 import Framework6


def get_framework(framework_id=None,
                  problem=None,
                  algorithm=None,
                  model_list=None,
                  ref_dirs=None,
                  curr_ref_id=None,
                  g_aggregate_func=None,
                  f_aggregate_func=None,
                  m5_fg_aggregate_func=None,
                  m6_fg_aggregate_func=None,
                  *args,
                  **kwargs):
    frameworks = framework_id.split(',')
    if len(frameworks) == 1:
        if framework_id in ['11']:
            return Framework11(framework_id=framework_id,
                               problem=problem,
                               algorithm=algorithm,
                               model_list=model_list,
                               ref_dirs=ref_dirs,
                               curr_ref_id=curr_ref_id,
                               g_aggregate_func=g_aggregate_func,
                               f_aggregate_func=f_aggregate_func,
                               m5_fg_aggregate_func=m5_fg_aggregate_func,
                               m6_fg_aggregate_func=m6_fg_aggregate_func,
                               *args, **kwargs)
        elif framework_id in ['12']:
            return Framework12(framework_id=framework_id,
                               problem=problem,
                               algorithm=algorithm,
                               model_list=model_list,
                               ref_dirs=ref_dirs,
                               curr_ref_id=curr_ref_id,
                               g_aggregate_func=g_aggregate_func,
                               f_aggregate_func=f_aggregate_func,
                               m5_fg_aggregate_func=m5_fg_aggregate_func,
                               m6_fg_aggregate_func=m6_fg_aggregate_func,
                               *args, **kwargs)
        elif framework_id in ['21']:
            return Framework21(framework_id=framework_id,
                               problem=problem,
                               algorithm=algorithm,
                               model_list=model_list,
                               ref_dirs=ref_dirs,
                               curr_ref_id=curr_ref_id,
                               g_aggregate_func=g_aggregate_func,
                               f_aggregate_func=f_aggregate_func,
                               m5_fg_aggregate_func=m5_fg_aggregate_func,
                               m6_fg_aggregate_func=m6_fg_aggregate_func,
                               *args, **kwargs)
        elif framework_id in ['22']:
            return Framework22(framework_id=framework_id,
                               problem=problem,
                               algorithm=algorithm,
                               model_list=model_list,
                               ref_dirs=ref_dirs,
                               curr_ref_id=curr_ref_id,
                               g_aggregate_func=g_aggregate_func,
                               f_aggregate_func=f_aggregate_func,
                               m5_fg_aggregate_func=m5_fg_aggregate_func,
                               m6_fg_aggregate_func=m6_fg_aggregate_func,
                               *args, **kwargs)
        elif framework_id in ['31']:
            return Framework31(framework_id=framework_id,
                               problem=problem,
                               algorithm=algorithm,
                               model_list=model_list,
                               ref_dirs=ref_dirs,
                               curr_ref_id=curr_ref_id,
                               g_aggregate_func=g_aggregate_func,
                               f_aggregate_func=f_aggregate_func,
                               m5_fg_aggregate_func=m5_fg_aggregate_func,
                               m6_fg_aggregate_func=m6_fg_aggregate_func,
                               *args, **kwargs)
        elif framework_id in ['32']:
            return Framework32(framework_id=framework_id,
                               problem=problem,
                               algorithm=algorithm,
                               model_list=model_list,
                               ref_dirs=ref_dirs,
                               curr_ref_id=curr_ref_id,
                               g_aggregate_func=g_aggregate_func,
                               f_aggregate_func=f_aggregate_func,
                               m5_fg_aggregate_func=m5_fg_aggregate_func,
                               m6_fg_aggregate_func=m6_fg_aggregate_func,
                               *args, **kwargs)
        elif framework_id in ['41']:
            return Framework41(framework_id=framework_id,
                               problem=problem,
                               algorithm=algorithm,
                               model_list=model_list,
                               ref_dirs=ref_dirs,
                               curr_ref_id=curr_ref_id,
                               g_aggregate_func=g_aggregate_func,
                               f_aggregate_func=f_aggregate_func,
                               m5_fg_aggregate_func=m5_fg_aggregate_func,
                               m6_fg_aggregate_func=m6_fg_aggregate_func,
                               *args, **kwargs)
        elif framework_id in ['42']:
            return Framework42(framework_id=framework_id,
                               problem=problem,
                               algorithm=algorithm,
                               model_list=model_list,
                               ref_dirs=ref_dirs,
                               curr_ref_id=curr_ref_id,
                               g_aggregate_func=g_aggregate_func,
                               f_aggregate_func=f_aggregate_func,
                               m5_fg_aggregate_func=m5_fg_aggregate_func,
                               m6_fg_aggregate_func=m6_fg_aggregate_func,
                               *args, **kwargs)
        elif framework_id in ['5']:
            return Framework5(framework_id=framework_id,
                              problem=problem,
                              algorithm=algorithm,
                              model_list=model_list,
                              ref_dirs=ref_dirs,
                              curr_ref_id=curr_ref_id,
                              g_aggregate_func=g_aggregate_func,
                              f_aggregate_func=f_aggregate_func,
                              m5_fg_aggregate_func=m5_fg_aggregate_func,
                              m6_fg_aggregate_func=m6_fg_aggregate_func,
                              *args, **kwargs)
        elif framework_id in ['6A', '6B']:
            return Framework6(framework_id=framework_id,
                              problem=problem,
                              algorithm=algorithm,
                              model_list=model_list,
                              ref_dirs=ref_dirs,
                              curr_ref_id=curr_ref_id,
                              g_aggregate_func=g_aggregate_func,
                              f_aggregate_func=f_aggregate_func,
                              m5_fg_aggregate_func=m5_fg_aggregate_func,
                              m6_fg_aggregate_func=m6_fg_aggregate_func,
                              *args, **kwargs)
        else:
            raise Exception("Framework definition not supported")
    else:
        raise Exception("Multiple Frameworks not supported")
