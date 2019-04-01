import numpy as np


def framework_candidate_select(framework_id, ref_dirs=None, pop=None, n_select=1):

    if n_select == 1:
        F = pop.get("F")
        if F.shape[1] > 1:  # if it is multi-objective, pick the middle one
            ref = (1 / F.shape[1]) * np.ones(F.shape[1])
            _F = np.sum(ref * F, 1)
            pop = pop[np.argmin(_F)]
        else:  # single-objective
            if np.any(pop.get("CV") <= 0):
                pop = pop[np.argmin(pop.get("F"))]
            else:
                pop = pop[np.argmin(pop.get("CV"))]
        return pop

    out_pop = []

    if framework_id in ['12', '22']:
        f = pop.get("F")
        f_asf = []
        if np.linalg.norm(np.max(f, axis=0) - np.min(f, axis=0)) > 1e-16:
            f_normalized = (f - np.min(f, axis=0)) / (np.max(f, axis=0) - np.min(f, axis=0))
        else:
            f_normalized = f

        for i in range(len(ref_dirs)):
            asf = np.max(f_normalized - ref_dirs[i], axis=1)
            f_asf.append(asf)

        f = np.column_stack(f_asf)
        cv = pop.get("CV")

    elif framework_id in ['32', '42']:
        f = pop.get("F")
        cv = pop.get("CV")
    else:
        f = pop.get("F")
        cv = np.zeros([f.shape[0], 1])

    feasible_index = (cv <= 0).flatten()  # find feasible index
    infeasible_index = (cv > 0).flatten()
    size = np.sum(infeasible_index)
    if size > 1:
        if np.sum(feasible_index) > 0:
            worst_of_all_ref_dir = np.max(f[feasible_index])
        else:
            worst_of_all_ref_dir = np.max(f)
        val_infeasible_sol = cv[infeasible_index] + worst_of_all_ref_dir
        f[infeasible_index, :] = np.tile(val_infeasible_sol, (1, f.shape[1]))

    for i in range(len(ref_dirs)):
        index = np.argsort(f[:, i])
        j = 0
        while index[j] in out_pop:
            j += 1
        out_pop.append(index[j])

    return pop[out_pop]
