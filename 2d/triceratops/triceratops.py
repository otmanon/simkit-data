

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
name = "triceratops"

png_path = dir + "/" + name + ".png"
image = Image.open(png_path)
X = gpt.png2poly(png_path)
V_list = []
E_list = []
indices = [0, 12, 14, 3, 4, 6, 2, 5,7, 8, 9, 10]
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

w_horn = 1 - winding_number(bc, V_list[1], E_list[1])
w_beak = 1 - winding_number(bc, V_list[2], E_list[2])

w_back = winding_number(bc, V_list[3], E_list[3])

w_rear_leg_0 = winding_number(bc, V_list[4], E_list[4])
w_rear_leg_1 = winding_number(bc, V_list[5], E_list[5])
w_rear_leg_2 = winding_number(bc, V_list[6], E_list[6])
w_rear_leg_3 = winding_number(bc, V_list[7], E_list[7])

w_front_leg_0 = winding_number(bc, V_list[8], E_list[8])
w_front_leg_1 = winding_number(bc, V_list[9], E_list[9])
w_front_leg_2 = winding_number(bc, V_list[10], E_list[10])
w_front_leg_3 = winding_number(bc, V_list[11], E_list[11])


igl.write_obj(dir +  "/" + name + ".obj", np.concatenate( [vertices, np.zeros((vertices.shape[0], 1))], axis=1), faces)
uv=vertices / [image.width, image.height]
np.save(dir + "/" + name + "_uv.npy", uv )

np.save(dir + "/" + name + "_w_horn.npy", w_horn )
np.save(dir + "/" + name + "_w_beak.npy", w_beak )
np.save(dir + "/" + name + "_w_back.npy", w_back )
np.save(dir + "/" + name + "_w_rear_leg_0.npy", w_rear_leg_0 )
np.save(dir + "/" + name + "_w_rear_leg_1.npy", w_rear_leg_1 )
np.save(dir + "/" + name + "_w_rear_leg_2.npy", w_rear_leg_2 )
np.save(dir + "/" + name + "_w_rear_leg_3.npy", w_rear_leg_3 )
np.save(dir + "/" + name + "_w_front_leg_0.npy", w_front_leg_0 )
np.save(dir + "/" + name + "_w_front_leg_1.npy", w_front_leg_1 )
np.save(dir + "/" + name + "_w_front_leg_2.npy", w_front_leg_2 )
np.save(dir + "/" + name + "_w_front_leg_3.npy", w_front_leg_3 )



mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=0)
mesh.add_parameterization_quantity("test_param",  uv,
                                defined_on='vertices')

mesh.add_scalar_quantity("w_horn", w_horn, defined_on='faces', cmap='blues', enabled=True)
mesh.add_scalar_quantity("w_beak", w_beak, defined_on='faces', cmap='blues', enabled=True)
mesh.add_scalar_quantity("w_back", w_back, defined_on='faces', cmap='blues', enabled=True)
mesh.add_scalar_quantity("w_rear_leg_0", w_rear_leg_0, defined_on='faces', cmap='blues', enabled=True)
mesh.add_scalar_quantity("w_rear_leg_1", w_rear_leg_1, defined_on='faces', cmap='blues', enabled=True)
mesh.add_scalar_quantity("w_rear_leg_2", w_rear_leg_2, defined_on='faces', cmap='blues', enabled=True)
mesh.add_scalar_quantity("w_rear_leg_3", w_rear_leg_3, defined_on='faces', cmap='blues', enabled=True)
mesh.add_scalar_quantity("w_front_leg_0", w_front_leg_0, defined_on='faces', cmap='blues', enabled=True)
mesh.add_scalar_quantity("w_front_leg_1", w_front_leg_1, defined_on='faces', cmap='blues', enabled=True)
mesh.add_scalar_quantity("w_front_leg_2", w_front_leg_2, defined_on='faces', cmap='blues', enabled=True)
mesh.add_scalar_quantity("w_front_leg_3", w_front_leg_3, defined_on='faces', cmap='blues', enabled=True)

vals = np.array(image)
mesh.add_color_quantity("test_vals", vals[:, :, 0:3]/255,
                        defined_on='texture', param_name="test_param",
                            enabled=True)
ps.show()

