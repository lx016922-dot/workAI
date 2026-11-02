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

A 392-mode candidate has also completed training on the same manufactured
samples. Its maximum transverse static response error is 4.9747%, vector error
1.7934%, and energy-norm error 15.9608%. The stricter static-coverage admission
still fails. Its complete source identity, states and report are retained under
`training/v12-nonlinear-basis392/`.

## Completed short waveform and physical checks

The complete trained spaces substantially improve the actual 20 us native
comparison. All directions, nonlinear differences and frequency bands are
included below; these are not vector-only scores.

| Complete candidate, 0.0125 us step | Maximum error | Material nonlinear error |
| --- | ---: | ---: |
| Trained 264-mode space | 1.9561% | 1.3511% |
| Trained 392-mode space | 1.6212% | 0.8078% |

Halving the candidate step from 0.025 to 0.0125 us changes the complete checks
by at most 0.3047% and 0.2753%, respectively. Comparing the two complete learned
spaces changes them by at most 1.7863%. The larger space contains the smaller
space to a maximum mass-norm residual of 1.292e-12; the same candidate weights,
training fields and graph are used. This is a check of nested trained spaces.

Keeping only the first 232 or 264 frequency coordinates of these richer spaces
still fails: those prefix changes reach 26.52% and 20.07%. The trained high
frequency directions therefore cannot be silently discarded. Both failed
prefix tests remain archived alongside the successful complete-space check.

Separate 64-field analytic admissions check 2,496 longitudinal/transverse
momentum and energy residuals per candidate. Maximum complete-graph residuals
are 0.03239% and 0.05976%, passing the fixed 2.5% physical limit. The alternative
Gauss-9 contraction differs by up to 17.0612% for the first candidate and is
not used in its prediction. These admissions do not establish quadrature or
full-duration accuracy.

`modal_nodal_force.py` preserves the same full graph operator while reconstructing
nodal displacements and assembling forces before modal projection. No quadrature
point or learned tensor entry is removed. Independent force/potential replay
differs by at most 3.606e-11; the actual 20 us trajectory differs by at most
7.128e-12 across the complete component/band checks. Its elapsed solve time is
252.9 seconds versus 544.4 seconds for the dense-gradient implementation under
the observed local load. This timing is not a general hardware guarantee.

All raw short predictions, coordinates, energies, identities, successful and
failed checks are under `diagnostics/v12-enriched-native-validation/`.

## Offline research package and next complete records

The original locally built 153,983,846-byte research archive includes the trained model,
training provenance, frozen inference code, all 20 complete 5 ms cases and a
network-free CLI. Every packaged file and archive readback has been verified.
It uses the pinned Python runtime from the existing offline distribution; it
does not need COMSOL, credentials or network access for prediction. The default
internal step was 0.05 us. Package identity is recorded in
`research/release-validation/v12-enriched-offline-local.json`.

Additional actual-step checks after resumption found that 0.05 us fails the
short native material/total nonlinear checks (5.2194% / 5.3150%) and the finer
time comparison (5.4532%). The replacement 0.025 us record passes all native
checks (maximum 1.9561%) and changes by at most 0.3047% on refinement to 0.0125 us.
The original coarse CI run was cancelled. The local coarse record is retained
only for research/time-refinement comparison.

The replacement 153,984,037-byte offline archive derives its default 0.025 us
step from a new frozen plan. All file hashes, archive readback and offline CLI
help were verified. Its identity is in
`research/release-validation/v12-enriched-step025-offline-local.json`.

New complete 5 ms linear-only diagnostics for both learned spaces pass 19/20
vector-waveform checks and 11/20 complete component/band checks. Worst relative
errors are 35.4677% for 264 modes and 40.1145% for 392 modes; unresolved reference
truncation/mesh failures prevent interpreting those references as converged
training labels. Full diagnostics and both successful/failed time-step checks
are retained in `diagnostics/v12-step-admission/`.

The dedicated replacement workflow first trains and admits the same research architecture,
including native and refinement checks at the actual long-run time step,
then runs all 20 complete 5 ms cases, independently audits their saved records,
and preserves distinct research assets. Short native success does not certify
the pending full 5 ms nonlinear comparisons. The default production weights
remain unchanged.
