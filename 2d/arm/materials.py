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


# winding_calf = np.load(data_dir + "/2d/" + character_name + "/leg_w_calf_muscle.npy").reshape(-1, 1)
# winding_tibia = np.load(data_dir + "/2d/" + character_name + "/leg_w_tibia_muscle.npy").reshape(-1, 1)
# winding_bot_thigh = np.load(data_dir + "/2d/" + character_name + "/leg_w_bot_thigh_muscle.npy").reshape(-1, 1)
# winding_top_thigh = np.load(data_dir + "/2d/" + character_name + "/leg_w_top_thigh_muscle.npy").reshape(-1, 1)

winding_bot_bone = np.load(data_dir + "/2d/" + character_name + "/arm_w_bone_1.npy").reshape(-1, 1)
winding_top_bone = np.load(data_dir + "/2d/" + character_name + "/arm_w_bone_2.npy").reshape(-1, 1)

inside_top = winding_top_bone < 0.1
inside_bot = winding_bot_bone < 0.1
inside_bone = np.logical_or(inside_top, inside_bot)

ym = np.ones((T.shape[0], 1), dtype=float) * 1e6
ym[inside_bone] = 1e14
pr = np.ones((T.shape[0], 1), dtype=float) * 0.49

path = data_dir + "/2d/" + character_name + "/" + character_name + "_materials.npy"

np.save(path, np.concatenate((ym, pr), axis=1))

import polyscope as ps

ps.init()

mesh = ps.register_surface_mesh(character_name, X, T)
mesh.add_scalar_quantity("ym", ym.flatten(), defined_on='faces', enabled=True)
ps.show()