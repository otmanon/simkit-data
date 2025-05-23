import os
import igl
import numpy as np


from simkit.pairwise_distance import pairwise_distance
from simkit.normalize_and_center import normalize_and_center

dir = os.path.dirname(__file__)

data_dir = dir 
mesh_file = data_dir + "/sea_turtle.mesh"
[X, T, F] = igl.read_mesh(mesh_file)
F = igl.boundary_facets(T)
X = normalize_and_center(X)
dim = 3


middle_vertex = np.array([[-0, 0.05, -0.2]])
D = pairwise_distance(middle_vertex, X)

pinned_epsilon = 0.2
pinned_vertices = np.where(D < pinned_epsilon)[1]

np.save(data_dir + "/sea_turtle_center_pinned_vertices.npy", pinned_vertices)

import polyscope as ps
ps.init()
ps.register_volume_mesh("octopus", X, T)
pc = ps.register_point_cloud("pinned", X[pinned_vertices], radius=0.01)
ps.show()

