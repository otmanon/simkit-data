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


character_name = "arm"

obj_path = data_dir + "/2d/" + character_name + "/" + character_name + ".obj"
[X, _, _, T, _, _] = igl.read_obj(obj_path)
X = X[:, :2]


winding = np.load(data_dir + "/2d/" + character_name + "/arm_w_muscle.npy").reshape(-1, 1)


path = data_dir + "/2d/" + character_name + "/" + character_name + "_muscle_fiber.npy"

muscle_fiber_builder_2D(X, T, winding > 0.5, path)
