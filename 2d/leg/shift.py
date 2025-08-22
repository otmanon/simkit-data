import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as psim
import os
import scipy as sp

import gpytoolbox as gpt


from simkit.apps.muscle_fiber_builder_2D import muscle_fiber_builder_2D
from simkit.diffuse_scalar import diffuse_scalar
from simkit.pairwise_distance import pairwise_distance
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix


dir = os.path.dirname(__file__)
data_dir = dir + "/../../../data/"


character_name = "arm"

obj_path = data_dir + "/2d/" + character_name + "/" + character_name + ".obj"
[X, _, _, T, _, _] = igl.read_obj(obj_path)
X = X[:, :2]
winding = np.load(data_dir + "/2d/" + character_name + "/w_muscle.npy")
mask = winding > 0.5

path = data_dir + "/2d/" + character_name + "/" + character_name + "_muscle_fiber.npy"

muscle_fiber_builder_2D(X, T, mask, path)
