"""Build the tendon-driven continuum arm and write it into the folder beside this file.

Not a mesh. A continuum arm with no continuum in it at all: thirteen stiff square plates
stacked up the z axis, each pair joined by four tendons and eight cross braces. Every
element is a spring, and the four tendon lines run twelve springs in series, so contracting
one line curls the whole arm towards it. Unlike `3d/octopus`, which is a tetrahedral body,
this is a spring network on its own and there is nothing to read with `igl`.

``octopus_arm.npz`` holds

    ``V``      (4*n_plates, 3) rest positions, plate `p` occupying rows `4p .. 4p+3`
    ``E``      (m, 2) springs
    ``group``  (m,) `0` plate, `1` brace, `2` tendon
    ``line``   (m,) which of the four tendon lines a tendon belongs to, `-1` otherwise
    ``Fq``     (2*n_plates, 3) triangles, two per plate, for drawing only
    ``pinned`` (4,) the base plate

Each plate is four vertices on the `+x, +y, -x, -y` axes of its own cross-section, so a
tendon line is named by the direction it curls the arm towards.

Standalone: numpy and nothing else.

Written for experiment 027 of the subspace_exploration repository, as `make_octopus_arm_3d`
in `experiments/027_mass_springs_demo/make_scenes.py`.

    python 3d/octopus_arm/octopus_arm.py
"""

import os

import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))


def make_octopus_arm(n_plates: int, segment: float, width_base: float,
                     width_tip: float) -> None:
    """Write `octopus_arm.npz`: an all-spring tendon-driven arm.

    Parameters
    ----------
    n_plates : int
        Plates stacked along z. `n_plates - 1` segments, so that many springs per line.
    segment : float
        Spacing between plates in metres, which is every tendon's rest length.
    width_base, width_tip : float
        Distance from the centre line to a plate vertex, at the base and at the tip. The
        taper is what makes it read as an arm rather than a column.
    """
    theta = np.arange(4) * (np.pi / 2.0)                       # +x, +y, -x, -y
    ring = np.column_stack([np.cos(theta), np.sin(theta)])
    w = np.linspace(width_base, width_tip, n_plates)
    z = np.arange(n_plates) * segment
    V = np.concatenate([np.column_stack([w[p] * ring, np.full(4, z[p])])
                        for p in range(n_plates)])

    E, group, line = [], [], []
    for p in range(n_plates):
        b = 4 * p
        for j in range(4):                                     # rim
            E.append((b + j, b + (j + 1) % 4)); group.append(0); line.append(-1)
        for j in range(2):                                     # the two diameters
            E.append((b + j, b + j + 2)); group.append(0); line.append(-1)
    for p in range(n_plates - 1):
        b, t = 4 * p, 4 * (p + 1)
        for j in range(4):                                     # tendons, straight up
            E.append((b + j, t + j)); group.append(2); line.append(j)
        for j in range(4):                                     # braces, both diagonals
            E.append((b + j, t + (j + 1) % 4)); group.append(1); line.append(-1)
            E.append((b + j, t + (j - 1) % 4)); group.append(1); line.append(-1)

    Fq = np.concatenate([np.array([[4 * p, 4 * p + 1, 4 * p + 2],
                                   [4 * p, 4 * p + 2, 4 * p + 3]])
                         for p in range(n_plates)])

    E = np.array(E)
    group = np.array(group)
    np.savez(os.path.join(DATA, "octopus_arm.npz"), V=V, E=E, group=group,
             line=np.array(line), Fq=Fq, pinned=np.arange(4))
    print(f"octopus_arm  {V.shape[0]:4d} vertices, {3 * V.shape[0]} dofs, "
          f"{E.shape[0]} springs "
          f"({int((group == 0).sum())} plate, {int((group == 1).sum())} brace, "
          f"{int((group == 2).sum())} tendon)")
    print(f"             {n_plates} plates, {n_plates - 1} springs in series per "
          f"tendon line, {z[-1]:.2f} m tall, "
          f"width {width_base:.3f} -> {width_tip:.3f} m")


if __name__ == "__main__":
    make_octopus_arm(n_plates=13, segment=0.07, width_base=0.11, width_tip=0.035)
    print(f"wrote into {DATA}")
