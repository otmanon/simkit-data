




import os
import igl
import numpy as np

from simkit.average_onto_simplex import average_onto_simplex
from simkit.pairwise_distance import pairwise_distance
from simkit.normalize_and_center import normalize_and_center
from simkit.winding_number import winding_number

dir = os.path.dirname(__file__)

data_dir = dir 
mesh_file = data_dir + "/pegasus.mesh"
[X, T, F] = igl.read_mesh(mesh_file)
F = igl.boundary_facets(T)
# X = normalize_and_center(X)
dim = 3


[Xs, _, _, Fs, _, _] = igl.read_obj(data_dir + "/pegasus_horn.obj")

BC = average_onto_simplex(X, T)
winding_numbers = igl.winding_number(Xs, Fs, BC)
winding_numbers = np.clip(winding_numbers, 0, 1).reshape(-1, 1)

inside_horn = winding_numbers > 0.5
outside_horn = winding_numbers <= 0.5

ym = np.ones((T.shape[0], 1)) * 1e6
ym[inside_horn] =  1e9

pr = np.ones((T.shape[0], 1)) * 0.

np.save(data_dir + "/pegasus_materials.npy", np.concatenate([ym, pr], axis=1))
# # num_contact_vertices = 100
# np.save(data_dir + "/pegasus_gmm_means.npy", gmm.means_)
# np.save(data_dir + "/pegasus_gmm_covariances.npy", gmm.covariances_)
import polyscope as ps
ps.init()
mesh = ps.register_volume_mesh("octopus", X, T)
ps.register_surface_mesh("horn", Xs, Fs)
mesh.add_scalar_quantity("winding_numbers", winding_numbers.flatten(), defined_on='cells', enabled=True)
mesh.add_scalar_quantity("ym", ym.flatten(), defined_on='cells', enabled=True)
ps.show()

