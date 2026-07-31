"""Build `teddy_bear.mesh`, the tetrahedral teddy, into the folder beside this file.

A 3D counterpart to the traced planar bear in `2d/teddy_bear`. It is not that outline
extruded or tetrahedralised; it is its own shape, built without a tet mesher.

An analytic signed distance field -- a smooth union of nine ellipsoids -- is sampled on a
regular grid, and every voxel whose centre is inside is cut into the six tetrahedra of the
Kuhn subdivision. That subdivision is conforming across a shared cube face, so the result
is a valid tet mesh with no cracks, and every element has exactly the same volume, so there
are no slivers to begin with.

What it costs is a staircase boundary. It is smoothed by fitting the mesh to the zero level
set -- minimise the Laplacian energy subject to a penalty pulling boundary vertices onto the
surface -- with each step backtracked until no tetrahedron has lost more than 65% of its
rest volume. That floor is what stops the fit: the staircase corners are the size of the
elements that would have to collapse to remove them, so the bear ends up rounded but still
visibly voxel-built. Raising `RES` is the way to make it smoother, at the cost of a slower
interactive solve.

    teddy_bear.mesh   MEDIT tet mesh, read back with igl.readMESH

Centred on the origin and scaled to 1 m tall. Standalone: numpy, scipy and libigl.

Written for experiment 024 of the subspace_exploration repository, as
`experiments/024_handle_demo/make_teddy_3d.py`.

    python 3d/teddy_bear/teddy_bear.py
"""

import os

import igl
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

DATA = os.path.dirname(os.path.abspath(__file__))

RES = 26              # voxels along the longest axis of the bounding box
SMOOTH_K = 0.10       # blend radius of the smooth union, in world units
N_FIT = 8             # smoothing / re-projection rounds
W_FIT = 1.0e3         # weight of the "sit on the level set" penalty against Laplacian energy
MIN_VOL_RATIO = 0.35  # a round is backtracked until every tet holds this share of its volume


# The bear: nine ellipsoids in a y-up frame, about 1.9 x 2.5 x 1.0 units before scaling.
PARTS = (
    ((0.00, 0.00, 0.00), (0.55, 0.68, 0.45)),      # body
    ((0.00, 0.95, 0.00), (0.42, 0.40, 0.40)),      # head
    ((0.00, 0.86, 0.34), (0.20, 0.17, 0.20)),      # snout
    ((-0.31, 1.26, 0.00), (0.17, 0.17, 0.12)),     # ears
    ((0.31, 1.26, 0.00), (0.17, 0.17, 0.12)),
    ((-0.64, 0.30, 0.00), (0.32, 0.19, 0.19)),     # arms
    ((0.64, 0.30, 0.00), (0.32, 0.19, 0.19)),
    ((-0.32, -0.74, 0.06), (0.25, 0.30, 0.26)),    # legs
    ((0.32, -0.74, 0.06), (0.25, 0.30, 0.26)),
)

# The six tetrahedra of the Kuhn subdivision of a unit cube, as corner bit patterns
# (i, j, k) -> i + 2j + 4k. Every one of them contains the 0-7 main diagonal, which is what
# makes the subdivision agree with its neighbour across a shared cube face.
KUHN = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7],
                 [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]])


def sdf(P: np.ndarray) -> np.ndarray:
    """Smooth-union signed distance to the teddy, negative inside.

    Each part contributes the ellipsoid distance ``|(p - c)/r| - 1`` rescaled by its
    smallest radius, the standard cheap stand-in for a true ellipsoid distance. The parts
    are combined with an exponential soft minimum, so the joins are filleted and the body
    comes out as one connected component.

    Parameters
    ----------
    P : np.ndarray (m, 3)
        Query points.

    Returns
    -------
    d : np.ndarray (m,)
        Signed distance, negative inside the bear.
    """
    d = np.full(P.shape[0], np.inf)
    for c, r in PARTS:
        c, r = np.asarray(c), np.asarray(r)
        di = (np.linalg.norm((P - c) / r, axis=1) - 1.0) * r.min()
        # Soft minimum, written so the exponential never overflows: min(a, b) - k log1p(...)
        d = np.minimum(d, di) - SMOOTH_K * np.log1p(np.exp(-np.abs(d - di) / SMOOTH_K))
    return d


def sdf_grad(P: np.ndarray, h: float = 1.0e-4) -> np.ndarray:
    """Central-difference gradient of :func:`sdf`, normalised to unit length.

    Parameters
    ----------
    P : np.ndarray (m, 3)
        Query points.
    h : float
        Finite-difference step, in world units.

    Returns
    -------
    g : np.ndarray (m, 3)
        Unit outward normals of the level sets through ``P``.
    """
    g = np.zeros_like(P)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        g[:, k] = sdf(P + e) - sdf(P - e)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1.0e-12)


def tet_volumes(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Signed volume of every tetrahedron.

    Parameters
    ----------
    V : np.ndarray (n, 3)
        Vertex positions.
    T : np.ndarray (m, 4)
        Tetrahedra.

    Returns
    -------
    vol : np.ndarray (m,)
        Signed volumes, positive for a positively oriented tetrahedron.
    """
    a, b, c, d = (V[T[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def voxel_tet_mesh(res: int) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralise the interior of :func:`sdf` on a regular grid.

    Keeps every voxel whose centre is inside, cuts each into the six Kuhn tetrahedra, drops
    the unreferenced grid nodes, and orients every element positively.

    Parameters
    ----------
    res : int
        Number of voxels along the longest axis of the bounding box.

    Returns
    -------
    V : np.ndarray (n, 3)
        Grid node positions.
    T : np.ndarray (m, 4)
        Tetrahedra, all of positive volume.
    """
    lo = np.min([np.asarray(c) - np.asarray(r) for c, r in PARTS], axis=0) - 0.05
    hi = np.max([np.asarray(c) + np.asarray(r) for c, r in PARTS], axis=0) + 0.05
    h = (hi - lo).max() / res
    n = np.ceil((hi - lo) / h).astype(int) + 1          # grid nodes per axis

    grid = [lo[k] + h * np.arange(n[k]) for k in range(3)]
    V = np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1).reshape(-1, 3)

    # One entry per voxel, indexed by its lower corner.
    C = np.stack(np.meshgrid(*[np.arange(n[k] - 1) for k in range(3)], indexing="ij"),
                 axis=-1).reshape(-1, 3)
    C = C[sdf(lo + h * (C + 0.5)) < 0.0]

    stride = np.array([n[1] * n[2], n[2], 1])
    corner = np.array([[b & 1, (b >> 1) & 1, (b >> 2) & 1] for b in range(8)])
    T = ((C[:, None, :] + corner[None, :, :]) @ stride)[:, KUHN].reshape(-1, 4)
    V, T, _, _ = igl.remove_unreferenced(V, T)

    flip = tet_volumes(V, T) < 0.0
    T[flip] = T[flip][:, [0, 1, 3, 2]]
    return np.ascontiguousarray(V), np.ascontiguousarray(T)


def fit_to_level_set(V: np.ndarray, T: np.ndarray, rounds: int) -> np.ndarray:
    """Round the staircase off by fitting the mesh to the zero level set of :func:`sdf`.

    Each round closest-point-projects the boundary onto the surface, then solves
    ``min ||L x||^2 + W_FIT ||x_b - p_b||^2`` over *all* vertices, so the interior absorbs
    part of the distortion instead of leaving it in the boundary layer. The move is then
    backtracked as a whole until every tetrahedron still holds ``MIN_VOL_RATIO`` of its rest
    volume, which is what keeps the mesh usable as a simulation mesh.

    Parameters
    ----------
    V : np.ndarray (n, 3)
        Vertex positions of the voxel mesh.
    T : np.ndarray (m, 4)
        Tetrahedra.
    rounds : int
        Number of project-then-smooth rounds.

    Returns
    -------
    V : np.ndarray (n, 3)
        Positions with the boundary rounded off, no element inverted.
    """
    n = V.shape[0]
    bI = np.unique(igl.boundary_facets(T)[0])
    vol0 = tet_volumes(V, T)

    # Graph Laplacian over the tet edges, and the boundary selection.
    E = np.vstack([T[:, [a, b]] for a in range(4) for b in range(4) if a < b])
    E = np.vstack([E, E[:, ::-1]])
    A = scipy.sparse.csr_matrix((np.ones(E.shape[0]), (E[:, 0], E[:, 1])), (n, n))
    A.data[:] = 1.0                                  # collapse duplicate edges to one
    L = scipy.sparse.diags(A.getnnz(axis=1).astype(float)) - A
    S = scipy.sparse.csr_matrix((np.ones(bI.size), (np.arange(bI.size), bI)), (bI.size, n))
    M = (L.T @ L + W_FIT * (S.T @ S)).tocsc()
    solve = scipy.sparse.linalg.factorized(M)

    V = V.copy()
    for _ in range(rounds):
        P = V[bI] - sdf(V[bI])[:, None] * sdf_grad(V[bI])
        step = np.column_stack([solve(W_FIT * (S.T @ P[:, k])) for k in range(3)]) - V
        alpha = 1.0
        for _ in range(25):
            W = V + alpha * step
            if (tet_volumes(W, T) / vol0).min() > MIN_VOL_RATIO:
                break
            alpha *= 0.6
        else:
            break
        V = W
    return V


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)

    V, T = voxel_tet_mesh(RES)
    print(f"voxel mesh   {V.shape[0]} vertices, {T.shape[0]} tets, "
          f"every element {tet_volumes(V, T).mean():.2e} in volume")

    V = fit_to_level_set(V, T, N_FIT)
    vol = tet_volumes(V, T)
    bI = np.unique(igl.boundary_facets(T)[0])
    print(f"fitted       max |sdf| on the boundary {np.abs(sdf(V[bI])).max():.3e}, "
          f"worst element at {(vol / tet_volumes(*voxel_tet_mesh(RES))).min():.2f} of rest")

    # Centre on the origin and scale the height to 1 m, so one set of material constants
    # means the same thing for the bear as it does for any other character at that scale.
    V = V - V.mean(axis=0)
    V = V / np.ptp(V[:, 1])
    print(f"final        {np.ptp(V, axis=0).round(3)} m bounding box, "
          f"{3 * V.shape[0]} degrees of freedom")

    igl.writeMESH(os.path.join(DATA, "teddy_bear.mesh"), V, T, igl.boundary_facets(T)[0])
    print("wrote", os.path.join(DATA, "teddy_bear.mesh"))
