

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
name = "jester"

full_png_path = dir + "/jester.png"
image = Image.open(full_png_path)

no_earrings_png_path = dir + "/jester_no_earrings.png"
X = gpt.png2poly(no_earrings_png_path)
V_list = []
E_list = []
indices = [0] # 2 is left earing, 17 is right earring 6 is face
for i in indices:
    # ps.register_point_cloud(str(i), X[i])
    x = X[i][:-1, :]
    E = closed_polyline(x)
    V_list.append(x)
    E_list.append(E)
    ps.register_curve_network("mesh_" + str(i), x, E)

ear_png_path = dir + "/jester_earrings.png"
X = gpt.png2poly(ear_png_path)
indices = [0, 2]
for i in indices:
    x = X[i][:-1, :]
    E = closed_polyline(x)
    V_list.append(x)
    E_list.append(E)
    ps.register_curve_network("mesh2_" + str(i), x, E)
ps.show()


# ps.show()
V_final, E_final = combine_meshes(V_list, E_list)
V_final, SVI, SVJ, _no = igl.remove_duplicate_vertices(V_final, E_final, 1.0)
E_final = SVJ[E_final]

holes = np.array([[1503., 449],
                [789, 429.]])
[vertices, faces] = igl.triangle.triangulate(V_final, E_final, flags="qVa400", H=holes)
bc = average_onto_simplex(vertices, faces)

w_ear_1 = np.abs(winding_number(bc, V_list[1], E_list[1]))
w_ear_2 = np.abs(winding_number(bc, V_list[2], E_list[2]))


inside_ear_1 = np.abs(w_ear_1) > 0.1
inside_ear_2 = np.abs(w_ear_2) > 0.1
inside_body = np.logical_not(np.logical_or(inside_ear_1, inside_ear_2))

labels = np.zeros((faces.shape[0]), dtype=int)
labels[inside_ear_1] = 0
labels[inside_ear_2] = 1
labels[inside_body] = 2

np.save(dir + "/" + name + "_labels.npy", labels)

igl.write_obj(dir +  "/" + name + ".obj", np.concatenate( [vertices, np.zeros((vertices.shape[0], 1))], axis=1), faces)
uv=vertices / [image.width, image.height]
np.save(dir + "/" + name + "_uv.npy", uv )

# np.save(dir + "/" + name + "_w_bone_1.npy", w_bone_1)
# np.save(dir + "/" + name + "_w_bone_2.npy", w_bone_2)
# np.save(dir + "/" + name + "_w_bone_3.npy", w_bone_3)
# np.save(dir + "/" + name + "_w_bone_4.npy", w_bone_4)

ym = np.ones((faces.shape[0], 1)) * 1e7
ym[w_ear_1 > 0.1] = 1e11
ym[w_ear_2 > 0.1] = 1e11
pr = np.ones((faces.shape[0], 1)) * 0.40
materials = np.concatenate([ym, pr], axis=1)
np.save(dir + "/" + name + "_materials.npy", materials)

middle = np.mean(vertices, axis=0)[None, :] - np.array([[-10, 400]])
pinned = np.where(np.linalg.norm(vertices - middle, axis=1) < 100)[0]
np.save(dir + "/" + name + "_pinned_vertices.npy", pinned)

mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=0)
mesh.add_parameterization_quantity("test_param",  uv,
                                defined_on='vertices')
mesh.add_scalar_quantity("top_bone", w_ear_1, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
mesh.add_scalar_quantity("bot_bone", w_ear_2, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
# mesh.add_scalar_quantity("top_thigh_muscle", w_face_3, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
# mesh.add_scalar_quantity("bot_thigh_muscle", w_bone_4, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
mesh.add_scalar_quantity("ym", ym.flatten(), defined_on='faces', enabled=True)

for i in range(3):
    mesh.add_scalar_quantity("face_distribution", labels, defined_on='faces', enabled=True, cmap='rainbow')
pc = ps.register_point_cloud("pinned", vertices[pinned], radius=0.01)

vals = np.array(image)
mesh.add_color_quantity("test_vals", vals[:, :, 0:3]/255,
                        defined_on='texture', param_name="test_param",
                            enabled=True)
ps.show()

