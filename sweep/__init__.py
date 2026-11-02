"""Parameter sweeps, training-data campaigns and mesh-convergence checks.

Engines: the plane-strain spectral solver (`2d_plane_strain`), the 3-D GLL spectral
solver (`3d_spectral`), the tetrahedral solver (`3d_tetra`) and the frozen graph
network through its version-2 input layer (`3d_graph`); `sweep.convergence` reruns
cases with a finer mesh, a higher degree and a smaller time step.

A sweep specification (`sweep.spec`) expands a base scene into immutable cases
(full-factorial grid, explicit overrides, Latin-hypercube or uniform sampling,
with linear links between fields). `sweep.runner` executes the cases in
isolated processes, resumably, and keeps a hash-verified manifest.
`sweep.summarize` extracts harmonics, accumulation fits, velocities and
amplitude-scaling checks; `sweep.dataset` stacks the receiver-line records of
all cases into one NPZ with geometry-grouped train/validation/test splits.

Nothing here trains a model or routes results to a surrogate.
"""
