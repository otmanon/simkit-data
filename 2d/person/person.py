"""Build the standing figure and its hair, into the folder beside this file.

* ``person.obj``       — head, neck, torso, two arms and two legs, 1.72 m tall
* ``person_hair.npz``  — six strands of mass springs, hanging past the shoulders

The figure is a union of axis-aligned rectangles meshed on a regular grid of square cells,
which is what keeps it free of pinch points -- a vertex where two otherwise separate patches
touch, which is a hinge the elastic energy cannot see and a static solve then rotates freely
about.

The hair is not part of the mesh. Each strand leaves the side of the head, arcs out past the
arm and hangs, sampled at equal arc length so every segment has the same rest length. Two
things hold a strand together: springs ``E`` between consecutive beads, which give it a
length, and bending hinges ``C``, triples ``(A, B, C)`` folding at ``B``, which give it a
shape. Springs alone would not do -- a spring at its rest length has stiffness only *along*
itself, so a straight chain of them has a singular Hessian sideways.

The first spring and the first two hinges reach back into the body, which is what anchors a
strand in direction as well as in position: a hinge folding at the root itself needs a
second body vertex to measure the fold against, and without it the whole strand can pivot
freely about its root and the static Hessian has one null direction per strand.

**Indices in ``E`` and ``C`` are into the concatenated state ``[person.obj vertices; hair
beads]``**, so the two files only mean anything together, and only at the resolution written
here.

``person_hair.npz`` holds ``V`` the beads, ``E`` the springs, ``C`` the hinges, ``roots``
the body vertex each strand grows from, and ``beads`` the bead count per strand.

Standalone: numpy and nothing else.

Written for experiment 028 of the subspace_exploration repository, as `make_person_2d` in
`experiments/028_inertia/make_scenes.py`.

    python 2d/person/person.py
"""

import os

import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))

#                 x0     x1     y0     y1
PERSON = ((-0.12,  0.12,  1.44,  1.72),      # head
          (-0.04,  0.04,  1.36,  1.44),      # neck
          (-0.20,  0.20,  0.88,  1.36),      # torso
          (-0.32, -0.20,  0.80,  1.32),      # arms
          ( 0.20,  0.32,  0.80,  1.32),
          (-0.16, -0.04,  0.00,  0.88),      # legs
          ( 0.04,  0.16,  0.00,  0.88))


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


def bezier_by_arclength(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                        n: int) -> np.ndarray:
    """`n` points equally spaced *along* the quadratic Bezier through `p0`, `p1`, `p2`.

    Equal spacing is the point: every segment of a strand then has the same rest length,
    so one number describes the whole chain.

    Parameters
    ----------
    p0, p1, p2 : np.ndarray (2,)
        Start, control and end point.
    n : int
        Number of samples, including both ends.

    Returns
    -------
    P : np.ndarray (n, 2)
        Samples, `P[0] = p0` and `P[-1] = p2`.
    """
    t = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    C = (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * p1 + t ** 2 * p2
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    q = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(q, s, C[:, 0]), np.interp(q, s, C[:, 1])])


def rects_to_mesh(rects: tuple, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate the union of axis-aligned rectangles on a regular grid of square cells.

    Every cell whose centre falls inside some rectangle is kept and cut into two triangles,
    and corners are shared between neighbours, so the result is one conforming mesh.

    Parameters
    ----------
    rects : tuple of (float, float, float, float)
        Rectangles as `(x0, x1, y0, y1)` in metres. Give bounds on the grid to avoid a
        staircase edge.
    cell : float
        Grid spacing in metres.

    Returns
    -------
    V : np.ndarray (n, 2)
        Vertex positions.
    F : np.ndarray (m, 3)
        Triangles, two per kept cell.
    """
    R = np.asarray(rects, dtype=float)
    lo = np.array([R[:, 0].min(), R[:, 2].min()])
    hi = np.array([R[:, 1].max(), R[:, 3].max()])
    nx, ny = (np.ceil((hi - lo) / cell)).astype(int) + 1
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    centre = lo + np.column_stack([i.ravel() + 0.5, j.ravel() + 0.5]) * cell
    inside = np.zeros(centre.shape[0], dtype=bool)
    for x0, x1, y0, y1 in R:
        inside |= ((centre[:, 0] > x0) & (centre[:, 0] < x1)
                   & (centre[:, 1] > y0) & (centre[:, 1] < y1))

    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b = idx[:-1, :-1].ravel()[inside], idx[1:, :-1].ravel()[inside]
    c, d = idx[1:, 1:].ravel()[inside], idx[:-1, 1:].ravel()[inside]
    F = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])

    X, Y = np.meshgrid(lo[0] + np.arange(nx) * cell, lo[1] + np.arange(ny) * cell,
                       indexing="ij")
    V = np.column_stack([X.ravel(), Y.ravel()])
    keep = np.unique(F)
    relabel = np.full(V.shape[0], -1)
    relabel[keep] = np.arange(keep.size)
    return V[keep], relabel[F]


def make_person(rects: tuple, cell: float, n_strands: int, n_seg: int,
                length: float, spread: float) -> None:
    """Write `person.obj` and `person_hair.npz`.

    Parameters
    ----------
    rects, cell
        Passed to :func:`rects_to_mesh`.
    n_strands : int
        Strands per side of the head.
    n_seg : int
        Segments per strand, so `n_seg + 1` beads before the one on the root is dropped.
    length : float
        Strand length in metres, measured along the curve.
    spread : float
        Distance from the centre line the strand hangs at, in metres. Wider than the
        shoulders keeps the hair outside the silhouette, where it can be seen moving.
    """
    V, F = rects_to_mesh(rects, cell)
    nv = V.shape[0]

    top = V[:, 1].max()
    half_head = V[V[:, 1] > top - 0.1, 0].max()

    beads, E, C, roots = [], [], [], []
    for side in (-1.0, 1.0):
        for k in range(n_strands):
            y = top - 2.0 * cell * k
            root = int(np.argmin(np.linalg.norm(V - [side * half_head, y], axis=1)))
            d_root = np.linalg.norm(V - V[root], axis=1)
            d_root[root] = np.inf
            scalp = int(np.argmin(d_root))     # the body vertex the fold is measured against
            p0 = V[root]
            # The sample sitting exactly on the root is dropped: a spring of zero rest
            # length has no direction, and its Hessian is `d/|d|` at `d = 0`.
            P = bezier_by_arclength(p0, np.array([p0[0], p0[1] - 0.5 * length]),
                                    np.array([side * spread, p0[1] - length]),
                                    n_seg + 1)[1:]
            base = nv + sum(b.shape[0] for b in beads)
            chain = [scalp, root] + [base + i for i in range(n_seg)]
            beads.append(P)
            roots.append(root)
            E += [(chain[i], chain[i + 1]) for i in range(1, n_seg + 1)]
            C += [(chain[i], chain[i + 1], chain[i + 2]) for i in range(n_seg)]

    Vh = np.concatenate(beads)
    np.savez(os.path.join(DATA, "person_hair.npz"), V=Vh, E=np.array(E),
             C=np.array(C), roots=np.array(roots), beads=np.array(n_seg))
    write_obj(os.path.join(DATA, "person.obj"), V, F)
    print(f"person    {nv:4d} vertices, {F.shape[0]:4d} triangles, "
          f"{np.ptp(V[:, 1]):.2f} m tall")
    print(f"  + hair  {Vh.shape[0]:4d} beads in {2 * n_strands} strands, "
          f"{len(E)} springs and {len(C)} bending hinges, "
          f"{length:.2f} m long, segment {length / n_seg:.3f} m")


if __name__ == "__main__":
    make_person(PERSON, cell=0.04, n_strands=3, n_seg=16, length=0.95, spread=0.26)
    print(f"wrote into {DATA}")
