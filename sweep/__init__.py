"""Parameter sweeps and training-data campaigns over the plane-strain solver.

A sweep specification (`sweep.spec`) expands a base scene into immutable cases
(full-factorial grid, explicit overrides, Latin-hypercube or uniform sampling,
with linear links between fields). `sweep.runner` executes the cases in
isolated processes, resumably, and keeps a hash-verified manifest.
`sweep.summarize` extracts harmonics, accumulation fits, velocities and
amplitude-scaling checks; `sweep.dataset` stacks the receiver-line records of
all cases into one NPZ with geometry-grouped train/validation/test splits.

Nothing here trains a model or routes results to a surrogate.
"""
