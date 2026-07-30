# Variation — j0001 (FOUND)

The base methodology, BUT instead of the single most obvious angle (pure
best-fit tightness), design the priority around the known item-size
distribution: items are uniform on (0.1, 0.7), so any residual capacity
in (0, 0.1) is permanently unfillable dead space. Score each feasible
bin by the usefulness of the residual the placement would leave,
penalizing placements in proportion to the dead capacity they lock in,
and calibrate the strength of that distribution-derived penalty
empirically rather than assuming tighter is always better.
