from pymoo.util.normalization import *


class NormalizeConstraint:

    def normalize(self, g, dataset_func=True):

        if dataset_func:
            f_normalized = (g - np.mean(g, axis=0)) / np.std(g, axis=0)
            # f_normalized = (g - np.min(g)) / (np.max(g) - np.min(g))
            return f_normalized
        else:
            neg_index = g <= 0
            pos_index = g > 0
            g_normalized = np.zeros(g.shape)
            for i in range(0, g.shape[1]):
                if g[neg_index[:, i], i].size > 1:  # normalize between -1 and 0
                    g_normalized[neg_index[:, i], i] = -1 + (g[neg_index[:, i], i] - np.min(g[neg_index[:, i], i]))/(np.max(g[neg_index[:, i], i])-np.min(g[neg_index[:, i], i]))
                    # g_normalized[neg_index[:, i], i] = -np.absolute((g[neg_index[:, i], i] - np.mean(g[neg_index[:, i], i], axis=0)) / np.std(g[neg_index[:, i], i], axis=0))
                else:
                    g_normalized[neg_index[:, i], i] = -1
                if g[pos_index[:, i], i].size > 1:  # normalize between 0 and 1
                    # g_normalized[pos_index[:, i], i] = (g[pos_index[:, i], i] - np.min(g[pos_index[:, i], i]))/(np.max(g[pos_index[:, i], i]) - np.min(g[pos_index[:, i], i]))
                    g_normalized[pos_index[:, i], i] = np.absolute((g[pos_index[:, i], i] - np.mean(g[pos_index[:, i], i], axis=0)) / np.std(g[pos_index[:, i], i], axis=0))
                else:
                    g_normalized[pos_index[:, i], i] = 1
            return g_normalized
