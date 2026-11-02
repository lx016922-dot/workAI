"""Version-2 input layer for the frozen graph network.

The frozen network (`graphwave`, offline 0.2.0/0.3.0) receives excitation only
through its external-force channel: one nodal load vector times one scalar
pulse, and whole-node fixity. This package widens that channel without
touching any frozen file or weight:

* any number of sources, each a nodal load (any planar face, region and
  spatial profile) times its own signal (every `physics.signals` type);
* component-wise constraints (fixed faces, symmetry, antisymmetry);
* a mass-proportional absorbing-layer field entering the state update.

The learned operators (derivative filters, energy network, state update and
time composition) are used exactly as frozen. Version-1 scenes are routed
to the frozen predictor unchanged.
"""
