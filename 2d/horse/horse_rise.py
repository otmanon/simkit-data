

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
from simkit.contact_springs_plane_gradient import contact_springs_plane_gradient
from simkit.massmatrix import massmatrix
from simkit.normal_force_matrix import normal_force_matrix
from simkit.simplex_vertex_averaging_matrix import simplex_vertex_averaging_matrix
from simkit.vertex_to_simplex_adjacency import vertex_to_simplex_adjacency
from simkit.volume import volume
from simkit.winding_number import winding_number
ps.init()

dir = os.path.dirname(__file__)
name = "horse_rise"

full_png_path = dir + "/horse_rise.png"
image = Image.open(full_png_path)

# no_earrings_png_path = dir + "/jester_no_earrings.png"
X = gpt.png2poly(full_png_path)
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


# ear_png_path = dir + "/jester_earrings.png"
# ps.show()

# ps.show()
V_final, E_final = combine_meshes(V_list, E_list)
V_final, SVI, SVJ, _no = igl.remove_duplicate_vertices(V_final, E_final, 10.0)
E_final = SVJ[E_final]

holes = np.array([]) #centroids)
#np.array([[1395.81, 850.64],
 #                 [825.99, 847.03]])
[vertices, faces] = igl.triangle.triangulate(V_final, E_final, flags="qVa300", H=holes)
[vertices, faces, _, _] = igl.remove_unreferenced(vertices, faces)
mesh = ps.register_surface_mesh("mesh", vertices, faces)
# ps.show()
bc = average_onto_simplex(vertices, faces)



igl.write_obj(dir +  "/" + name + ".obj", np.concatenate( [vertices, np.zeros((vertices.shape[0], 1))], axis=1), faces)
uv=vertices / [image.width, image.height]
np.save(dir + "/" + name + "_uv.npy", uv )
import polyscope as ps

ps.init()


mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=0)
mesh.add_parameterization_quantity("test_param",  uv,
                                defined_on='vertices')

vals = np.array(image)
mesh.add_color_quantity("test_vals", vals[:, :, 0:3]/255,
                        defined_on='texture', param_name="test_param",
                            enabled=True)
ps.show()

