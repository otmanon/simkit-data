import igl
import numpy as np

[X, T, _] = igl.readMESH("data/3d/crab/crab.mesh")

jointI = igl.readDMAT("data/3d/crab/stiffI.dmat")

# cI = igl.readDMAT("data/3d/crab/cI.dmat")
fixI = X[:, 2] < np.min(X[:, 2]) + 0.01 
    # V(:,3) < min(V(:, 3)) + 0.01;
np.save("data/3d/crab/is_joint.npy", jointI)
np.save("data/3d/crab/bI.npy", np.where(fixI)[0])

mu = jointI * 1e6 + np.logical_not(jointI)*1e12
np.save("data/3d/crab/mu.npy", mu)

import polyscope as ps
ps.init()
pc = ps.register_point_cloud("pc", X[fixI.astype(int).flatten(), :])

mesh = ps.register_volume_mesh("crab", X, T)
mesh.add_scalar_quantity("stiffI", jointI.flatten(), defined_on="cells")
mesh.add_scalar_quantity("mu", mu.flatten(), defined_on="cells")
ps.show()
