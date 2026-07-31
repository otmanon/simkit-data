"""Build the tendon-driven planar arm and write it into the folder beside this file.

An arm that articulates rather than bows, driven by an antagonistic pair of tendons rather
than by a muscle field. Distinct from `2d/arm`, which is png-traced with rigid bones, and
from `2d/soft_arm`, which is uniformly soft: here the stiffness is what carves out a joint.

* ``tendon_arm.obj``           — the triangulated arm, 1.0 m long, shoulder at ``x = 0``
* ``tendon_arm_tendons.npy``   — ``(2, 2)`` vertex index pairs; row 0 flexor, row 1 extensor
* ``tendon_arm_stiffness.npy`` — ``(m, 1)`` per-triangle multiplier on Young's modulus

The arm is a grid warped by a thickness profile: a bicep bulge, a waist at the elbow, a
forearm and a narrow wrist. The stiffness field is ``1`` in the bones and
``joint_softness`` in the elbow band, which is the only thing that makes the arm articulate
rather than bow like a beam. The flexor runs over the top and the extensor under the bottom,
both attached at 0.18 and 0.82 along the arm, so the two fight each other.

Both index arrays are into ``tendon_arm.obj`` as written here; regenerate at a different
``nx``, ``ny`` and they no longer mean anything.

Standalone: numpy and nothing else.

Written for experiment 027 of the subspace_exploration repository, as `make_arm_2d` in
`experiments/027_mass_springs_demo/make_scenes.py`.

    python 2d/tendon_arm/tendon_arm.py
"""

import os

import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))


def grid_mesh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right-triangle mesh of the rectangle spanned by the 1D coordinate arrays `x`, `y`.

    Vertex `i * y.size + j` sits at `(x[i], y[j])`, so a row or column of the grid is a
    contiguous slice and picking out an edge of the rectangle is one comparison.

    Parameters
    ----------
    x : np.ndarray (nx,)
        Coordinates along the first axis, increasing.
    y : np.ndarray (ny,)
        Coordinates along the second axis, increasing.

    Returns
    -------
    V : np.ndarray (nx*ny, 2)
        Vertex positions.
    F : np.ndarray (2*(nx-1)*(ny-1), 3)
        Triangles, counter-clockwise.
    """
    X, Y = np.meshgrid(x, y, indexing="ij")
    V = np.column_stack([X.ravel(), Y.ravel()])
    idx = np.arange(x.size * y.size).reshape(x.size, y.size)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return V, F


def write_obj(path: str, V: np.ndarray, F: np.ndarray) -> None:
    """Write a planar mesh as an OBJ with `z = 0`, so libigl can read it back.

    Parameters
    ----------
    path : str
        File to write, overwritten if it exists.
    V : np.ndarray (n, 2)
        Vertex positions.
    F : np.ndarray (m, 3)
        Triangles, zero indexed; OBJ is one indexed and the shift happens here.
    """
    with open(path, "w") as handle:
        for v in V:
            handle.write(f"v {v[0]:.9f} {v[1]:.9f} 0.0\n")
        for f in F:
            handle.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def make_tendon_arm(nx: int, ny: int, joint_half_width: float,
                    joint_softness: float) -> None:
    """Write `tendon_arm.obj`, `tendon_arm_tendons.npy` and `tendon_arm_stiffness.npy`.

    Parameters
    ----------
    nx, ny : int
        Vertices along and across the arm.
    joint_half_width : float
        Half width of the soft elbow band in metres, measured along x from the elbow.
    joint_softness : float
        Young's modulus multiplier inside that band, between 0 and 1.
    """
    # Half thickness against arc length: shoulder, bicep, elbow, forearm, wrist.
    profile_x = np.array([0.00, 0.22, 0.50, 0.72, 1.00])
    profile_h = np.array([0.100, 0.115, 0.055, 0.085, 0.045])

    x = np.linspace(0.0, 1.0, nx)
    V, F = grid_mesh(x, np.linspace(-1.0, 1.0, ny))
    V[:, 1] *= np.interp(V[:, 0], profile_x, profile_h)

    top = np.where(V[:, 1] > 0.0)[0][np.argsort(V[np.where(V[:, 1] > 0.0)[0], 0])]
    bot = np.where(V[:, 1] < 0.0)[0][np.argsort(V[np.where(V[:, 1] < 0.0)[0], 0])]
    top = top[np.isclose(V[top, 1], np.interp(V[top, 0], profile_x, profile_h))]
    bot = bot[np.isclose(V[bot, 1], -np.interp(V[bot, 0], profile_x, profile_h))]

    def span(row: np.ndarray) -> list[int]:
        """The two vertices of `row` nearest 0.18 and 0.82 along the arm."""
        return [int(row[np.argmin(np.abs(V[row, 0] - s))]) for s in (0.18, 0.82)]

    tendons = np.array([span(top), span(bot)])        # row 0 flexor, row 1 extensor

    centroid = V[F].mean(axis=1)
    stiffness = np.where(np.abs(centroid[:, 0] - 0.5) < joint_half_width,
                         joint_softness, 1.0).reshape(-1, 1)

    write_obj(os.path.join(DATA, "tendon_arm.obj"), V, F)
    np.save(os.path.join(DATA, "tendon_arm_tendons.npy"), tendons)
    np.save(os.path.join(DATA, "tendon_arm_stiffness.npy"), stiffness)
    print(f"tendon_arm  {V.shape[0]:4d} vertices, {F.shape[0]:4d} triangles, "
          f"2 tendons, {int((stiffness < 1.0).sum())} soft elbow triangles")
    print(f"            flexor   {V[tendons[0, 0]].round(3)} -> "
          f"{V[tendons[0, 1]].round(3)}, a0 = "
          f"{np.linalg.norm(V[tendons[0, 1]] - V[tendons[0, 0]]):.3f} m")
    print(f"            extensor {V[tendons[1, 0]].round(3)} -> "
          f"{V[tendons[1, 1]].round(3)}, a0 = "
          f"{np.linalg.norm(V[tendons[1, 1]] - V[tendons[1, 0]]):.3f} m")


if __name__ == "__main__":
    make_tendon_arm(nx=41, ny=7, joint_half_width=0.075, joint_softness=0.02)
    print(f"wrote into {DATA}")
