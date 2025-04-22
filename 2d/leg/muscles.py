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


winding_calf = np.load(data_dir + "/2d/" + character_name + "/leg_w_calf_muscle.npy").reshape(-1, 1)
winding_tibia = np.load(data_dir + "/2d/" + character_name + "/leg_w_tibia_muscle.npy").reshape(-1, 1)
winding_bot_thigh = np.load(data_dir + "/2d/" + character_name + "/leg_w_bot_thigh_muscle.npy").reshape(-1, 1)
winding_top_thigh = np.load(data_dir + "/2d/" + character_name + "/leg_w_top_thigh_muscle.npy").reshape(-1, 1)

winding = np.concatenate((winding_calf, winding_tibia, winding_bot_thigh, winding_top_thigh), axis=1)

path = data_dir + "/2d/" + character_name + "/" + character_name + "_muscle_fiber.npy"

muscle_fiber_builder_2D(X, T, winding > 0.1, path)
