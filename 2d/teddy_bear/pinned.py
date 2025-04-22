import numpy as np
import os
import igl

from simkit import common_selections
from simkit.normalize_and_center import normalize_and_center


data_dir = os.path.dirname(__file__)
character_name = "teddy_bear"
[X, _, _, T, _, _] = igl.read_obj(data_dir + "/"+ character_name + ".obj")
X = X[:, :2]    
X = normalize_and_center(X)
_, pinned_vertices = common_selections.center_indices(X, 0.1)
np.save(data_dir + "/"+ character_name + "_pinned_vertices.npy", pinned_vertices)
  
  
import polyscope as ps
ps.init()
ps.register_surface_mesh(character_name, X, T)
ps.register_point_cloud("pinned", X[pinned_vertices], radius=0.01)
ps.show()
