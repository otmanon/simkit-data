

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
name = "arm"

png_path = dir + "/" + name + ".png"
image = Image.open(png_path)
X = gpt.png2poly(png_path)
V_list = []
E_list = []
indices = [0, 2, 5, 4]  # 0 outline, 9 top bone, 4 bot bone, 11 top thigh muscle, 13 bot thigh muscle, 3 tibia muscle, 12 calf msucle
for i in indices:
    # ps.register_point_cloud(str(i), X[i])
    x = X[i][:-1, :]
    E = closed_polyline(x)
    V_list.append(x)
    E_list.append(E)
    ps.register_curve_network("mesh_" + str(i), x, E)

ps.show()
V_final, E_final = combine_meshes(V_list, E_list)
V_final, SVI, SVJ, _no = igl.remove_duplicate_vertices(V_final, E_final, 5.0)
E_final = SVJ[E_final]
[vertices, faces] = igl.triangle.triangulate(V_final, E_final, flags="qVa400")
bc = average_onto_simplex(vertices, faces)

w_bone_1 = 1 - winding_number(bc, V_list[1], E_list[1])
w_bone_2 = 1 - winding_number(bc, V_list[2], E_list[2])

w_muscle = winding_number(bc, V_list[3], E_list[3])


igl.write_obj(dir +  "/" + name + ".obj", np.concatenate( [vertices, np.zeros((vertices.shape[0], 1))], axis=1), faces)
uv=vertices / [image.width, image.height]
np.save(dir + "/" + name + "_uv.npy", uv )

np.save(dir + "/" + name + "_w_bone_1.npy", w_bone_1)
np.save(dir + "/" + name + "_w_bone_2.npy", w_bone_2)
np.save(dir + "/" + name + "_w_muscle.npy", w_muscle)


mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=0)
mesh.add_parameterization_quantity("test_param",  uv,
                                defined_on='vertices')
mesh.add_scalar_quantity("top_bone", w_bone_1, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
mesh.add_scalar_quantity("bot_bone", w_bone_2, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
mesh.add_scalar_quantity("top_thigh_muscle", w_muscle, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))

vals = np.array(image)
mesh.add_color_quantity("test_vals", vals[:, :, 0:3]/255,
                        defined_on='texture', param_name="test_param",
                            enabled=True)
ps.show()

