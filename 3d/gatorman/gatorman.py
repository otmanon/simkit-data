import igl
import numpy as np

[X, T, _] = igl.readMESH("data/3d/gatorman/gatorman.mesh")

in_stiff = igl.readDMAT("data/3d/gatorman/material.dmat")

mu = in_stiff * 1e12 + (1 - in_stiff) * 1e5
# cI = igl.readDMAT("data/3d/crab/cI.dmat")
fixI = igl.readDMAT("data/3d/gatorman/fixIn.DMAT")

np.save("data/3d/gatorman/mu.npy", mu)

    # V(:,3) < min(V(:, 3)) + 0.01;
# np.save("data/3d/gatorman/is_joint.npy", jointI)
np.save("data/3d/gatorman/bI.npy", np.where(fixI)[0])

tail_ind = np.where(X[:, 2] < np.min(X[:, 2]) + 1)[0]
np.save("data/3d/gatorman/pullI.npy", tail_ind)
import polyscope as ps
ps.init()
# pc = ps.register_point_cloud("pc", X[fixI.astype(bool).flatten(), :])
# pc2 = ps.register_point_cloud("pc2", X[tail_ind.astype(bool).flatten(), :])
mesh = ps.register_volume_mesh("crab", X, T)
# mesh.add_scalar_quantity("stiffI", material.flatten(), defined_on="cells")
mesh.add_scalar_quantity("mu", mu.flatten(), defined_on="cells")
ps.show()
