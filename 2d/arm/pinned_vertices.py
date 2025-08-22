import igl
import os
import numpy as np
from simkit.normalize_and_center import normalize_and_center

data_dir = os.path.dirname(__file__) 
[X, _, _, T, _, _] = igl.read_obj(data_dir + "/arm.obj")
X = normalize_and_center(X)

top_vertices = X[:, 1] > X[:, 1].max() - 0.1

np.save(data_dir + "/arm_pinned_vertices.npy", top_vertices)
import polyscope as ps

ps.init()
mesh = ps.register_surface_mesh("arm", X, T)
pc = ps.register_point_cloud("top", X[top_vertices], radius=0.01)
ps.show()
