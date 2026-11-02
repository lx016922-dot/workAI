"""Learned coarse-graph operators for the frozen graph network (direction B).

The frozen network's element operators equal the exact GLL spectral operators,
so its accuracy and cost are those of a spectral solver. This package learns
replacement element operators (derivative filter, stiffness quadrature, mass
weights) under hard consistency constraints, so that a coarser graph with
fewer nodes and layers reproduces converged fine-grid waveforms and second
harmonics. The energy network and state update stay frozen.
"""
