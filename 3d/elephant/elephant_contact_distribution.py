import os
import igl
import numpy as np
from sklearn.mixture import GaussianMixture


from simkit.gaussian_pdf import gaussian_pdf
from simkit.pairwise_distance import pairwise_distance
from simkit.normalize_and_center import normalize_and_center

dir = os.path.dirname(__file__)

data_dir = dir 
mesh_file = data_dir + "/elephant.mesh"
[X, T, F] = igl.read_mesh(mesh_file)
F = igl.boundary_facets(T)
X = normalize_and_center(X)
dim = 3


# let's fit a gaussian mixture model to the vertices X
gmm = GaussianMixture(20, covariance_type='diag')
gmm.fit(X)


# num_contact_vertices = 100
np.save(data_dir + "/elephant_gmm_means.npy", gmm.means_)
np.save(data_dir + "/elephant_gmm_covariances.npy", gmm.covariances_)
import polyscope as ps
ps.init()
mesh = ps.register_volume_mesh("octopus", X, T)
# pc = ps.register_point_cloud("pinned", X[pinned_vertices], radius=0.01)
ps.load_color_map("data_cmap", data_dir + "/../../colormaps/Purples_11.png")
means = ps.register_point_cloud("means", gmm.means_, radius=0.05)
for i in range(20):
    pdf = gaussian_pdf(X , gmm.means_[i], np.diag(gmm.covariances_[i]))
    mesh.add_scalar_quantity("pdf" + str(i), pdf, enabled=True, cmap="data_cmap")
ps.show()

