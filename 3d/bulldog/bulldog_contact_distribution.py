

import os
import igl
import numpy as np
from sklearn.mixture import GaussianMixture


from simkit.pairwise_distance import pairwise_distance
from simkit.normalize_and_center import normalize_and_center

dir = os.path.dirname(__file__)

data_dir = dir 
mesh_file = data_dir + "/bulldog.mesh"
[X, T, F] = igl.read_mesh(mesh_file)
F = igl.boundary_facets(T)
X = normalize_and_center(X)
dim = 3


# let's fit a gaussian mixture model to the vertices X
gmm = GaussianMixture(10, covariance_type='diag')
gmm.fit(X)


# num_contact_vertices = 100
np.save(data_dir + "/bulldog_gmm_means.npy", gmm.means_)
np.save(data_dir + "/bulldog_gmm_covariances.npy", gmm.covariances_)
import polyscope as ps
ps.init()
ps.register_volume_mesh("octopus", X, T)
# pc = ps.register_point_cloud("pinned", X[pinned_vertices], radius=0.01)

means = ps.register_point_cloud("means", gmm.means_, radius=0.01)
ps.show()

