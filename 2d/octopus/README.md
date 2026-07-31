# octopus (2d)

A planar octopus character: a round head with four tapered tentacles, lined with muscle
springs. The 3D `3d/octopus` is a different asset — a tetrahedral body with no springs — and
this one is not a slice of it.

    octopus.obj           2100 vertices, 3725 triangles, spanning 1.80 m by 1.21 m
    octopus_springs.npy   (88, 2) vertex index pairs, one row per muscle spring

Each tentacle is lined by two chains of springs, one down each side, attached at points
jittered along the tentacle so that no chain is a straight line of collinear vertices. The
springs carry **no labels** — nothing in the file says which tentacle a spring belongs to or
which side of it. That is deliberate: the experiment this was built for has to recover the
grouping from the physics.

`octopus_springs.npy` indexes `octopus.obj` as shipped. Re-meshing invalidates it.

## Provenance

Built for experiment 020 of the `subspace_exploration` repository by
`experiments/020_soft_actuated_character/make_octopus_character.py`.

**There is no build script in this folder.** The geometry lives in that repository's library
— `octopus_2d_character` lays the outline out, `polygon_mesh` triangulates it,
`strip_spring_edges` lines each tentacle — and copying those four hundred lines here would
only give this repo a second copy to drift out of step with. Run the generator there to
rebuild; the numbers that make *this* octopus are:

| | |
|---|---|
| head radius | 0.35 m, centred on the origin |
| tentacles | 4, leaving at −150°, −110°, −70°, −30° |
| tentacle curl | −48°, −20°, 20°, 48° over their length |
| tentacle length | 0.60 m |
| tentacle half width | 0.075 m at the head, 0.34 of that at the tip |
| taper exponent | 2.5, so nearly straight sided until the last third |
| samples | 40 centreline points per tentacle, 14 boundary points per head arc |
| element area | 3.0e-4 m², about five elements across a tentacle |
| muscles | 12 attachments per side per tentacle, so 11 springs |
| muscle span | 0.10 to 0.88 along the tentacle, off the rounded tip |
| muscle offset | 0.45 to 0.85 of the local half width |
| muscle jitter | 0.30 of the attachment spacing |
| seed | 20 for the first tentacle, incrementing, so adding one does not reshuffle the others |
