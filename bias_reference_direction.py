from pymoo.util.reference_direction import ReferenceDirectionFactory, \
    UniformReferenceDirectionFactory, MultiLayerReferenceDirectionFactory
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



class BiasReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self, n_dim, scaling=None, n_points=None, n_partitions=None, bias=None) -> None:

        super().__init__(n_dim, scaling=scaling)
        self.n_partitions = n_partitions
        self.n_dim = n_dim
        self.scaling = scaling
        self.n_points= n_points

        if bias is not None:
            self.seq = np.hstack((bias * np.ones(self.n_dim - 1), 1))
        else:
            self.seq = np.hstack((10*np.ones(self.n_dim-1), 1))

    def _do(self):
        ref_dirs =  MultiLayerReferenceDirectionFactory([
            UniformReferenceDirectionFactory(n_dim=self.n_dim, n_partitions=12, scaling=1),
            UniformReferenceDirectionFactory(n_dim=self.n_dim, n_partitions=12, scaling=0.7)]).do()

        #UniformReferenceDirectionFactory(n_dim=self.n_dim,n_points=self.n_points, n_partitions=self.n_partitions, scaling=self.scaling).do()
        self.n_points = ref_dirs.shape[0]
        self.seq = np.tile(self.seq, [ref_dirs.shape[0], 1])
        ref_dirs = np.multiply(ref_dirs, self.seq)
        sum = np.sum(ref_dirs, axis=1)
        sum = np.tile(sum, [ref_dirs.shape[1], 1]).transpose()
        ref_dirs = np.divide(ref_dirs, sum)
        return ref_dirs


aux_ref_dirs = BiasReferenceDirectionFactory(3, n_points=100,  bias=10).do()
# print(aux_ref_dirs)
# x = np.logspace(0, 1, 3, endpoint=True)
# y = np.linspace(0, 1, 10, endpoint=True)
# z = np.linspace(0, 1, 10, endpoint=True)
# X, Y, Z = np.meshgrid(x, y, z)
# aux_ref_dirs = np.concatenate((np.vstack(X.flatten()), np.vstack(Y.flatten()), np.vstack(Z.flatten())), axis=1)
#
# aux_ref_dirs = aux_ref_dirs/np.vstack(np.sum(aux_ref_dirs, axis=1))
# aux_ref_dirs =  np.unique(aux_ref_dirs, axis=0)
print(aux_ref_dirs.shape)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(aux_ref_dirs[:, 0], aux_ref_dirs[:, 1], aux_ref_dirs[:, 2], color='r')
plt.show()
