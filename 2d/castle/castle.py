

import os
import polyscope as ps
import gpytoolbox as gpt
import numpy as np
from PIL import Image
import igl

import igl.triangle
from simkit.closed_polyline import closed_polyline
from simkit.combine_meshes import combine_meshes

dir = os.path.dirname(__file__)
name = "castle"

png_path = dir + "/" + name + ".png"
image = Image.open(png_path)
X = gpt.png2poly(png_path)
V_list = []
E_list = []
indices = [0] #range(len(X))
for i in indices:
    # ps.register_point_cloud(str(i), X[i])
    x = X[i][:-1, :]
    E = closed_polyline(x)
    V_list.append(x)
    E_list.append(E)


V_final, E_final = combine_meshes(V_list, E_list)

V_final, SVI, SVJ, _no = igl.remove_duplicate_vertices(V_final, E_final, 30.0)
E_final = SVJ[E_final]

[vertices, faces] = igl.triangle.triangulate(V_final, E_final, flags="qVa200")


igl.write_obj(dir +  "/" + name + ".obj", np.concatenate( [vertices, np.zeros((vertices.shape[0], 1))], axis=1), faces)
uv=vertices / [image.width, image.height]
np.save(dir + "/" + name + "_uv.npy", uv )
ps.init()
mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=0)
mesh.add_parameterization_quantity("test_param",  uv,
                                defined_on='vertices')
vals = np.array(image)
mesh.add_color_quantity("test_vals", vals[:, :, 0:3]/255,
                        defined_on='texture', param_name="test_param",
                            enabled=True)
# ps.register_curve_network("heart", V_final, E_final)
ps.show()

