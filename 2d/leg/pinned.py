import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as psim
import os
import scipy as sp

import gpytoolbox as gpt


from simkit.apps.muscle_fiber_builder_2D import muscle_fiber_builder_2D



dir = os.path.dirname(__file__)
data_dir = dir + "/../../../data/"


character_name = "leg"

obj_path = data_dir + "/2d/" + character_name + "/" + character_name + ".obj"
[X, _, _, T, _, _] = igl.read_obj(obj_path)
X = X[:, :2]


#rightmost vertices
rightmost = X[:, 0] >  X[:, 0].max() - 30

pinned_indices = np.where(rightmost)[0]

path = data_dir + "/2d/" + character_name + "/" + character_name + "_pinned.npy"

np.save(path, pinned_indices)


import polyscope as ps
ps.init()
ps.register_surface_mesh("leg", X, T)
ps.register_point_cloud("pinned", X[pinned_indices, :])
ps.show()
