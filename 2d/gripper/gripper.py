

import os

import polyscope as ps
import gpytoolbox as gpt
import numpy as np
from PIL import Image
import igl

import igl.triangle


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../..")))
from simkit.average_onto_simplex import average_onto_simplex
from simkit.boundary_loops import boundary_loops
from simkit.closed_polyline import closed_polyline
from simkit.combine_meshes import combine_meshes
from simkit.contact_springs_plane_gradient import contact_springs_plane_gradient
from simkit.massmatrix import massmatrix
from simkit.normal_force_matrix import normal_force_matrix
from simkit.simplex_vertex_averaging_matrix import simplex_vertex_averaging_matrix
from simkit.vertex_to_simplex_adjacency import vertex_to_simplex_adjacency
from simkit.volume import volume
from simkit.winding_number import winding_number
ps.init()

dir = os.path.dirname(__file__)
name = "gripper"

full_png_path = dir + "/gripper.png"
image = Image.open(full_png_path)

# no_earrings_png_path = dir + "/jester_no_earrings.png"
X = gpt.png2poly(full_png_path)
V_list = []
E_list = []
indices = [0, 3, 5, 6, 7, 8, 9, 26, 27, 28, 29, 30, 31] # 
# indices = range(len(X))
for i in indices:
    # ps.register_point_cloud(str(i), X[i])
    x = X[i][:-1, :]
    E = closed_polyline(x)
    V_list.append(x)
    E_list.append(E)
    ps.register_curve_network("mesh_" + str(i), x, E)

centroids = []
hole_indices_in_indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).flatten()
hole_indices = np.array(indices)[hole_indices_in_indices]
for i in hole_indices:
    x = X[i][:-1, :]
    E = closed_polyline(x)
    V_list.append(x)
    E_list.append(E)
    
    centroid = np.mean(x, axis=0)
    centroids.append(centroid)



# ps.show()
# ps.init()
# ps.remove_all_structures()
stiff_png_path = dir + "/gripper_stiff.png"
X = gpt.png2poly(stiff_png_path)
indices = [0]
# indices = range(len(X))
for i in indices:
    x = X[i][:-1, :]
    E = closed_polyline(x)
    V_list.append(x)
    E_list.append(E)
    ps.register_curve_network("mesh2_" + str(i), x, E)
# ps.show()


V_final, E_final = combine_meshes(V_list, E_list)
V_final, SVI, SVJ, _no = igl.remove_duplicate_vertices(V_final, E_final, 5.0)
E_final = SVJ[E_final]

holes = np.array(centroids)

# ps.remove_all_structures()
# ps.init()

# # ps.register_curve_network("mesh", V_final, E_final)
# # ps.show()
# ps.remove_all_structures()
#np.array([[1395.81, 850.64],
 #                 [825.99, 847.03]])
# [vertices, faces] = igl.triangle.triangulate(V_final, E_final, flags="qVa200", H=holes)
# [vertices, faces, _, _] = igl.remove_unreferenced(vertices, faces)
# mesh = ps.register_surface_mesh("mesh", vertices, faces)
# ps.show()
# ps.show()
V_final, E_final = combine_meshes(V_list, E_list)
V_final, SVI, SVJ, _no = igl.remove_duplicate_vertices(V_final, E_final, 10.0)
E_final = SVJ[E_final]

holes = np.array(centroids)
#np.array([[1395.81, 850.64],
 #                 [825.99, 847.03]])
[vertices, faces, _, _, _] = igl.triangle.triangulate(V_final, E_final, flags="qVa300", H=holes)
[vertices, faces, _, _] = igl.remove_unreferenced(vertices, faces)


igl.writeOBJ(dir +  "/" + name + ".obj", np.concatenate( [vertices, np.zeros((vertices.shape[0], 1))], axis=1), faces)
uv=vertices / [image.width, image.height]
np.save(dir + "/" + name + "_uv.npy", uv )



mesh = ps.register_surface_mesh("mesh", vertices, faces)

Es = boundary_loops(faces, vertices.shape[0])



E = np.vstack(Es[1:])


bc = average_onto_simplex(vertices, faces)

w_stiff = np.abs(winding_number(bc, V_list[-1], E_list[-1]))
# w_actuator = np.abs(winding_number(bc, V_list[2], E_list[2])) + np.abs(winding_number(bc, V_list[3], E_list[3]))
inside_stiff = np.abs(w_stiff) > 0.1

ym = np.ones((faces.shape[0], 1)) * 5e7
ym[inside_stiff > 0.1] = 1e9
# ym[inside_actuator > 0.1] = 0
pr = np.ones((faces.shape[0], 1)) * 0.45
# pr[inside_actuator > 0.1] = 0.0
materials = np.concatenate([ym, pr], axis=1)


# now let's build the actuator force distribution
# bI = np.where(labels == 1)[0]
N = normal_force_matrix(vertices, E)
X = vertices
Tan = X[E[:, 1], :] - X[E[:, 0], :]
Tan = Tan / np.linalg.norm(Tan, axis=1)[:, None]
N = Tan.copy()
N[:, 0] = Tan[:, 1]
N[:, 1] = -Tan[:, 0]

length = volume(X, E)

A = simplex_vertex_averaging_matrix(E, X.shape[0], length)
Nv  = A @ N
Nv_norm = np.linalg.norm(Nv, axis=1)
Nv[Nv_norm > 1e-6] = Nv[Nv_norm > 1e-6] / Nv_norm[Nv_norm > 1e-6, None]

M = massmatrix(X, E)
f = M @ Nv 

np.save(dir + "/" + name + "_force_E.npy", E)
np.save(dir + "/" + name + "_force_Es.npy", np.array(Es, dtype=object))
np.save(dir + "/" + name + "_force.npy", f)
import polyscope as ps

ps.init()
ps.remove_all_structures()
mesh = ps.register_curve_network("curve", vertices, E)
mesh.add_vector_quantity("normal", N, defined_on='edges')
mesh.add_vector_quantity("normal_vert", Nv)
# ps.show()

np.save(dir + "/" + name + "_materials.npy", materials)

top = vertices[:, 1].max() #- np.array([[-10, 400]])
pinned = np.where(top - vertices[:,1] < 160)[0]
np.save(dir + "/" + name + "_pinned_vertices.npy", pinned)
mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=0)
mesh.add_parameterization_quantity("test_param",  uv,
                                defined_on='vertices')
mesh.add_scalar_quantity("stiff", w_stiff, defined_on='faces', enabled=True, vminmax=np.array([0, 1.]))
mesh.add_scalar_quantity("ym", ym.flatten(), defined_on='faces', enabled=True)
pc = ps.register_point_cloud("pinned", vertices[pinned], radius=0.01)
vals = np.array(image)
mesh.add_color_quantity("test_vals", vals[:, :, 0:3]/255,
                        defined_on='texture', param_name="test_param",
                            enabled=True, filter_mode='nearest')
ps.show()

