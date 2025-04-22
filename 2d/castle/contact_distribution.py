import igl
import polyscope as ps
import os
import numpy as np
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.force_dual_modes import force_dual_modes_sqrt
from simkit.linear_elasticity_hessian import linear_elasticity_hessian
from simkit.massmatrix import massmatrix
from simkit.pairwise_distance import pairwise_distance
from simkit.normalize_and_center import normalize_and_center
from simkit.polyscope.view_displacement_modes import view_displacement_modes
from simkit.polyscope.view_scalar_fields import view_scalar_fields
import scipy as sp
dir = os.path.dirname(__file__)
[X, _, _, T, _, _] = igl.read_obj(dir + "/castle.obj")
X = X[:, :2]
X = normalize_and_center(X)


bI = np.unique(igl.boundary_facets(T))
left_side_indices =(X[bI,0] < 0)
thresh = 0.5
mid_top = X[bI, 1] < np.mean(X[bI, 1])+ thresh
mid_bot = X[bI, 1] > np.mean(X[bI, 1])- thresh
mid = np.logical_and(mid_top, mid_bot)

indices = np.logical_and(left_side_indices, mid)

cI = bI[indices]


threshold = 7.5e-1
D = pairwise_distance(X[cI], X)
# D_within = D < threshold
D_within = np.exp(-0.0001*D)

M = massmatrix(X, T)
M_sqrt = sp.sparse.diags(np.sqrt(M.diagonal()))
M_sqrt_e = sp.sparse.kron(M_sqrt, sp.sparse.eye(2))
sigma_F_sqrt = D_within.T *1

sigma_F_sqrt = M_sqrt @ sigma_F_sqrt 

sigma_F_sqrt_e = np.kron(sigma_F_sqrt, np.eye(2))
H = linear_elasticity_hessian(X=X, T=T, ym=1e6, pr=0.45)

pinned = X[:, 1] < np.min(X[:, 1]) + 0.1
pinned_vertices = np.where(pinned)[0]
H_pin = dirichlet_penalty(pinned_vertices, X[pinned_vertices], X.shape[0], 1e8)[0]

H = H + H_pin

B, d = force_dual_modes_sqrt(H, sigma_F_sqrt_e, 10, M_sqrt_e)

view_displacement_modes(X, T, B,a=0.05, period=20)
view_scalar_fields(X, T, )

import polyscope as ps

ps.init()
ps.register_surface_mesh("mesh", X, T)
ps.register_point_cloud("interst", X[cI], radius=0.01)
ps.show()