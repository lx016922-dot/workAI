"""Plane-strain (x–z section) nonlinear spectral element solver.

The constitutive law is the three-dimensional Murnaghan / StVK / linear
energy of `spectral3d.solver.constitutive`, evaluated on the displacement
gradient embedded in three dimensions with u_y = 0 and no y-dependence.
Forces, masses and energies are per metre of out-of-plane width.
"""
