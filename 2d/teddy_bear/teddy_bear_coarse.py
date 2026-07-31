"""Build `teddy_bear_coarse.obj` from `teddy_bear.obj`, in the folder beside this file.

The traced bear is 4584 vertices, which is more than a dynamics demo needs. Decimating a
triangulation is fiddlier than throwing it away: take the boundary loop, keep every
`STRIDE`-th vertex of it, and re-triangulate that outline under an area bound. The result is
679 vertices, blockier in silhouette, centred on the origin and scaled to 1 m tall rather
than left in the source's pixel units.

Nothing in `teddy_bear.obj` survives the round trip, so the pinned selection in
`teddy_bear_pinned_vertices.npy` does not index this mesh.

Standalone: numpy and libigl, including `igl.triangle`.

Written for experiment 028 of the subspace_exploration repository, as `make_teddy_2d` in
`experiments/028_inertia/make_scenes.py`.

    python 2d/teddy_bear/teddy_bear_coarse.py
"""

import os

import igl
import igl.triangle
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(DATA, "teddy_bear.obj")

STRIDE = 6            # keep every STRIDE-th vertex of the source boundary loop
MAX_AREA = 700.0      # largest triangle allowed, in the source's own pixel units
HEIGHT = 1.0          # metres, the height of the finished bear


def make_teddy_coarse(stride: int, max_area: float, height: float) -> None:
    """Write `teddy_bear_coarse.obj`: the traced bear, coarsened and scaled.

    Parameters
    ----------
    stride : int
        Keep every `stride`-th vertex of the source boundary loop. Larger is a blockier
        silhouette.
    max_area : float
        Largest triangle area allowed, in the source's own pixel units.
    height : float
        Height of the finished bear, in metres.
    """
    V, F = igl.read_triangle_mesh(SRC)
    outline = V[igl.boundary_loop(F)][::stride, :2]
    n = outline.shape[0]
    segments = np.column_stack([np.arange(n), (np.arange(n) + 1) % n])
    V, F = igl.triangle.triangulate(outline, segments, flags=f"qQa{max_area}")[:2]

    V = V - V.mean(axis=0)
    V = V * (height / np.ptp(V[:, 1]))
    igl.write_triangle_mesh(os.path.join(DATA, "teddy_bear_coarse.obj"),
                            np.hstack([V, np.zeros((V.shape[0], 1))]), F)
    print(f"teddy_bear_coarse  {V.shape[0]:4d} vertices, {F.shape[0]:4d} triangles, "
          f"{np.ptp(V[:, 0]):.2f} m by {np.ptp(V[:, 1]):.2f} m, "
          f"from {n} outline points")


if __name__ == "__main__":
    make_teddy_coarse(STRIDE, MAX_AREA, HEIGHT)
    print(f"wrote into {DATA}")
