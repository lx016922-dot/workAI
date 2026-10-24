"""Guided-wave toolchain for the nonlinear Lamb-wave workflow.

* `lamb`: Rayleigh–Lamb dispersion (phase and group velocity, mode shapes) of a free isotropic plate;
* `matching`: synchronous (phase-matched) mode pairs for second-harmonic generation;
* `design`: wavelength, comb pitch, wedge angle, time windows and symmetry hints for a chosen mode;
* `analysis`: exact-frequency harmonic coefficients, accumulation fits, amplitude scaling,
  phase/group velocity from a receiver line, f–k spectra and through-thickness mode decomposition.

Everything here is analytic or signal processing; nothing runs a solver or a neural network.
"""
