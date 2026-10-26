# Continued nonlinear training and complete-record evidence

The requested **all-check error below 5% is not yet established**. The default
offline weights remain unchanged. This continuation uses the same material
tensors, complete graph quadrature, three-dimensional coupled momentum
equations and conservative energy.

## Completed 5 ms records and offline reproduction

All Hann, Gaussian, rectangular and chirp cases with 1, 3, 5, 10 and 20 cycles
completed: 20 records, each with 25,001 samples and linear, geometric-nonlinear
and complete Murnaghan controls. Prediction run 40204192440 is frozen at
`1cb35bc82c056b9960f3e94d4f7c679afe8f2ee6`.

Audit run 40286412270 independently checked saved coordinates, reconstructed
receiver displacements, sampled potential energy, complete time grids, source
identities and every recorded component/band diagnostic. It published four
signal evidence archives and a small audit archive to release v0.3.0. The first
audit failure and its numerical replay repair remain documented separately;
the scientific comparison functions and 5% threshold were not changed.

The independent **linear COMSOL-derived** checks pass in 19/20 cases for vector
waveforms and 10/20 cases for all components and frequency bands. Several
reference modal/mesh convergence checks also fail. These are diagnostic results,
not a full-duration nonlinear COMSOL certificate. All records include the
actual nonlinear candidate histories; those histories still need an independent
converged full 5 ms nonlinear reference.

The complete local Hann one-cycle record, produced with the published phase
model and pinned offline runtime, matches the corresponding cloud record:
maximum relative error `1.1248808854837656e-10`, maximum absolute RMS difference
`1.2247289955896516e-22 m`. Every control, nonlinear difference, component and
frequency-band replay check passes the separate reproducibility limits.

Evidence:
- `diagnostics/v12-complete-matrix-audit/`: all 20 audits and release identities.
- `diagnostics/v12-complete-matrix-figures/`: source-backed CSV, PNG and PDF.
- `diagnostics/v12-offline5ms-replay/`: complete offline/cloud comparison.

## Native COMSOL convergence

Both 20 us native refinement jobs are complete and their artifacts are verified
against signed manifests. Their original degree, step and export guards remain
unchanged.

| Check | Maximum component/band change |
| --- | ---: |
| h = 0.75 to 0.50 mm | 0.8362% |
| Native step 0.01 to 0.005 us | 0.2197% |
| Unchanged 500-mode candidate material increment against the finer native record | 29.2749% |

The last row remains a failure. The converged short native checks do not extend
to 5 ms. The user confirmed that other tasks currently share the remote COMSOL
server, whose research template has changed. This task has not replaced that
template or submitted its prepared nodal jobs. Local training and comparison
against already archived native results continue.

## Nonlinear basis training

`tools/train-modal-enrichment.py` learns displacement directions from candidate
physics alone. For each manufactured state it separates geometric/material
nonlinear forcing and excites each x/y/z momentum equation. Each response solves
the complete coupled candidate tangent; no transverse coupling is dropped.
Mass-normalized POD gives equal weight to these responses, then a Rayleigh-Ritz
rotation retains the exact projected quadratic energy. Every original graph
integration point and material tensor entry remains in the nonlinear predictor.

No COMSOL trajectory, validation waveform, fitted constitutive correction or
online static solver enters the trained predictor. The static solves are used
only to generate training examples. Fixed/symmetry constraints, orthonormality,
positive stiffness, strain, Jacobian and force residual guards are checked.

The first candidate adds 64 directions to the published 200-mode basis. It uses
48 manufactured training states and 64 separate holdout states, with maximum
gradient norms between 2e-5 and 8e-4. The worst holdout transverse mass-norm
response error falls from 16.0840% to 7.6038%; vector error falls from 6.5896% to
2.9957%. The stricter 2.5% static-coverage target fails, including an energy-norm
error of 23.4683%. These are static representation metrics, not waveform errors.
The complete executed source, states and failed admission are retained under
`training/v12-nonlinear-basis264/`.

A 392-mode candidate is being trained on the same manufactured samples to
measure the effect of a larger learned space. Independent analytic checks and
short native comparisons follow before any inference promotion. Neither
candidate currently supports a new accuracy claim.
