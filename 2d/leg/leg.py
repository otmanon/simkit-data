

import os
import polyscope as ps
import gpytoolbox as gpt
import numpy as np
from PIL import Image
import igl

import igl.triangle
from simkit.average_onto_simplex import average_onto_simplex
from simkit.closed_polyline import closed_polyline
from simkit.combine_meshes import combine_meshes
from simkit.winding_number import winding_number
ps.init()

dir = os.path.dirname(__file__)
name = "leg"

png_path = dir + "/" + name + ".png"
image = Image.open(png_path)
X = gpt.png2poly(png_path)
V_list = []
E_list = []
indices = [0, 9, 4, 11, 13, 3, 12]  # 0 outline, 9 top bone, 4 bot bone, 11 top thigh muscle, 13 bot thigh muscle, 3 tibia muscle, 12 calf msucle
for i in indices:
    # ps.register_point_cloud(str(i), X[i])
    x = X[i][:-1, :]
    E = closed_polyline(x)
    V_list.append(x)
    E_list.append(E)
    ps.register_curve_network("mesh_" + str(i), x, E)


V_final, E_final = combine_meshes(V_list, E_list)
V_final, SVI, SVJ, _no = igl.remove_duplicate_vertices(V_final, E_final, 10.0)
E_final = SVJ[E_final]
[vertices, faces] = igl.triangle.triangulate(V_final, E_final, flags="qVa100")
bc = average_onto_simplex(vertices, faces)

w_top_bone = 1 - winding_number(bc, V_list[1], E_list[1])
w_bot_bone = 1 - winding_number(bc, V_list[2], E_list[2])

w_top_thigh_muscle = winding_number(bc, V_list[3], E_list[3])
w_bot_thigh_muscle = winding_number(bc, V_list[4], E_list[4])

w_tibia_muscle = winding_number(bc, V_list[5], E_list[5])
w_calf_muscle = winding_number(bc, V_list[6], E_list[6])


igl.write_obj(dir +  "/" + name + ".obj", np.concatenate( [vertices, np.zeros((vertices.shape[0], 1))], axis=1), faces)
uv=vertices / [image.width, image.height]
np.save(dir + "/" + name + "_uv.npy", uv )

np.save(dir + "/" + name + "_w_top_bone.npy", w_top_bone)
np.save(dir + "/" + name + "_w_bot_bone.npy", w_bot_bone)
np.save(dir + "/" + name + "_w_top_thigh_muscle.npy", w_top_thigh_muscle)
np.save(dir + "/" + name + "_w_bot_thigh_muscle.npy", w_bot_thigh_muscle)
np.save(dir + "/" + name + "_w_tibia_muscle.npy", w_tibia_muscle)
np.save(dir + "/" + name + "_w_calf_muscle.npy", w_calf_muscle)


mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=0)
mesh.add_parameterization_quantity("test_param",  uv,
                                defined_on='vertices')
mesh.add_scalar_quantity("top_bone", w_top_bone, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
mesh.add_scalar_quantity("bot_bone", w_bot_bone, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
mesh.add_scalar_quantity("top_thigh_muscle", w_top_thigh_muscle, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
mesh.add_scalar_quantity("bot_thigh_muscle", w_bot_thigh_muscle, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
mesh.add_scalar_quantity("tibia_muscle", w_tibia_muscle, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
mesh.add_scalar_quantity("calf_muscle", w_calf_muscle, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))

vals = np.array(image)
mesh.add_color_quantity("test_vals", vals[:, :, 0:3]/255,
                        defined_on='texture', param_name="test_param",
                            enabled=True)
ps.show()

