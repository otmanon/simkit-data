"""Build the soft planar arm and write it into the folder beside this file.

A smooth limb with no bones in it, unlike the traced `2d/arm`, whose two rigid bones and
png-traced silhouette make it a different character: this one is meant to be actuated by
muscle fields over a soft body.

* ``soft_arm.obj``            — the triangulated arm, shoulder at the origin
* ``soft_arm_fine.obj``       — the same arm at four times the element count
* ``soft_arm_centerline.npy`` — ``(N, 3)``, the centreline ``x, y`` and its radius

The arm is an offset curve: a centreline from the shoulder through the elbow and the
wrist to the fingertips, and a radius profile along it that bulges at the bicep and pinches
at the elbow and the wrist. Offsetting the centreline by that radius on both sides gives a
closed outline, and Triangle fills it.

The muscles are *not* built here, and this script does not write them. They are a field over
the elements rather than a property of the geometry, built off the centreline, which is why
the centreline is written out beside the mesh. That is also why both resolutions can share
one muscle definition: the field is defined against the limb, not against the elements. The
four fields shipped in ``soft_arm_muscles.npy`` — ``(4, 808, 3)``, one row per element
holding the fibre direction scaled by the activation weight, named in
``soft_arm_muscle_names.npy`` as bicep, tricep, forearm and wrist — index ``soft_arm.obj``
and came out of the notebook named below, not out of this file.

Standalone: numpy, libigl and nothing else.

Written for experiment 026 of the subspace_exploration repository, as
`experiments/026_emu_actuation/make_arm.py`; the muscle fields come from
`experiments/026_emu_actuation/026_emu_actuation.py`.

    python 2d/soft_arm/soft_arm.py
"""

import os

import numpy as np

import igl
import igl.triangle

OUT = os.path.dirname(os.path.abspath(__file__))

# Centreline joints in metres: shoulder, elbow, wrist, fingertip. A nearly straight arm,
# because a straight rest pose is what makes the flexion of each muscle legible.
JOINTS = np.array([[0.00, 0.000],
                   [0.34, 0.000],
                   [0.60, -0.150],
                   [0.71, -0.225]])
# Radius at a few fractions of the arclength: deltoid, bicep bulge, elbow, forearm belly,
# wrist, hand, fingertip.
R_KNOTS = np.array([[0.00, 0.085], [0.20, 0.098], [0.46, 0.046], [0.64, 0.058],
                    [0.82, 0.026], [0.89, 0.054], [0.96, 0.046], [1.00, 0.018]])
N_CENTER = 90           # samples along the centreline
# The centreline is smoothed much harder than the radius. Rounding the elbow is what stops
# the inner offset curve from crossing itself -- the turn has to bend over an arc longer
# than the radius there -- while the radius has to keep the notch at the wrist sharp.
SMOOTH_C, SMOOTH_R = 5.0, 1.5     # samples
N_CAP = 8               # points in the round cap at the fingertip
AREA = 2.0e-4           # m^2, upper bound on a triangle: about 700 elements


def smooth(A, sigma):
    """Gaussian-smooth the rows of `A` along axis 0, holding the ends fixed."""
    k = np.exp(-0.5 * (np.arange(-3 * int(sigma), 3 * int(sigma) + 1) / sigma) ** 2)
    k /= k.sum()
    pad = (k.size - 1) // 2
    P = np.vstack([np.repeat(A[:1], pad, axis=0), A, np.repeat(A[-1:], pad, axis=0)])
    return np.stack([np.convolve(P[:, j], k, mode="valid") for j in range(A.shape[1])], 1)


def centerline():
    """Sample the centreline and its radius: `(C, r, tangent, normal)`, each `(N_CENTER, .)`.

    The polyline through the joints is resampled uniformly in arclength and then smoothed,
    which rounds off the two corners at the elbow and the wrist — a kinked centreline would
    make its own offset curve cross itself.
    """
    seg = np.linalg.norm(np.diff(JOINTS, axis=0), axis=1)
    s_j = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, s_j[-1], N_CENTER)
    C = smooth(np.stack([np.interp(s, s_j, JOINTS[:, 0]),
                         np.interp(s, s_j, JOINTS[:, 1])], 1), SMOOTH_C)

    # Re-parameterise by the arclength of the smoothed curve, so `u` is what the radius
    # knots and the muscle windows in the notebook are quoted against.
    d = np.linalg.norm(np.diff(C, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    u /= u[-1]
    r = smooth(np.interp(u, R_KNOTS[:, 0], R_KNOTS[:, 1])[:, None], SMOOTH_R)[:, 0]

    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.stack([-T[:, 1], T[:, 0]], 1)              # left normal, +y at the shoulder
    return C, r, u, T, N


def outline(C, r, T, N):
    """The closed boundary polygon: up one side, round the fingertip, back down the other.

    The shoulder end is cut flat, which is where the arm is pinned.
    """
    top, bot = C + r[:, None] * N, C - r[:, None] * N
    ang = np.linspace(-np.pi / 2, np.pi / 2, N_CAP + 2)[1:-1]
    cap = C[-1] + r[-1] * (np.cos(ang)[:, None] * T[-1] + np.sin(ang)[:, None] * N[-1])
    return np.vstack([top, cap[::-1], bot[::-1]])


def mesh_at(P, S, area):
    """Triangulate the outline with an upper bound of `area` on any element."""
    # The flag string is parsed by Triangle itself and it stops at the 'e' of an exponent,
    # so the area bound has to be written out positionally.
    flag = np.format_float_positional(area, unique=True, trim="0")
    V, F, _, _, _ = igl.triangle.triangulate(P, S, flags=f"qa{flag}Q")
    return np.ascontiguousarray(V[:, :2]), np.ascontiguousarray(F)


def main():
    C, r, u, T, N = centerline()
    P = outline(C, r, T, N)
    S = np.stack([np.arange(P.shape[0]), np.roll(np.arange(P.shape[0]), -1)], 1)

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "soft_arm_centerline.npy"), np.hstack([C, r[:, None]]))
    print(f"outline  {P.shape[0]} boundary points, radius {r.min():.3f} to {r.max():.3f} m, "
          f"centreline {C.shape[0]} samples")

    for name, area in (("soft_arm.obj", AREA), ("soft_arm_fine.obj", AREA / 4.0)):
        V, F = mesh_at(P, S, area)
        igl.write_triangle_mesh(os.path.join(OUT, name),
                                np.hstack([V, np.zeros((V.shape[0], 1))]), F)
        edge = np.linalg.norm(V[F[:, [1, 2, 0]]] - V[F], axis=2)
        e1, e2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
        print(f"{name:13s} {V.shape[0]:5d} vertices, {F.shape[0]:5d} triangles, "
              f"{np.ptp(V[:, 0]):.3f} x {np.ptp(V[:, 1]):.3f} m, area "
              f"{0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum():.4f} m^2, "
              f"edges {edge.min():.4f} to {edge.max():.4f} m")
    print(f"wrote    {OUT}")


if __name__ == "__main__":
    main()
