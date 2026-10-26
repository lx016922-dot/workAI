# Resumed computation and actual-step admission

The resumed 20-case research workflow initially selected a 0.05 us internal
step. An additional actual prediction at that step failed the existing native
20 us comparison: material increment 5.2194%, total nonlinear increment 5.3150%.
Its difference from the 0.0125 us prediction reached 5.4532%. The failed raw
record and metrics are retained in `native-step050/` and `time050-vs0125.json`.

The replacement 0.025 us prediction passes all native waveform, component and
frequency-band checks, with a maximum of 1.9561%. Its difference from the
0.0125 us prediction is at most 0.3047%. Its raw record and checks are retained
in `native-step025/` and `time025-vs0125.json`. Both comparisons use the same
complete 264-mode candidate and the same nodal force implementation.

The finer 0.0125 us record remains in the adjacent
`v12-enriched-native-validation/nodal264-native/` archive. Its file identities
are recorded in both time comparisons. The scientific 5% and small-signal
absolute gates are unchanged. Two time steps alone do not establish an observed
convergence order or a complete 5 ms nonlinear accuracy certificate.

Run 40309994368 at commit bee6331 was cancelled before its 20-case long matrix.
The identity-checked cancellation completed in run 40311520221. The local coarse
5 ms calculation remains a research record for later time refinement; it is
not an admitted default. The replacement plan is
`plans/v12-enriched-nodal-step025.json`; its workflow admits the actual 0.025 us
step and the 0.0125 us comparison before starting the full matrix. Its distinct
offline package derives the default step from the frozen plan.

## Complete-record linear diagnosis

`linear-phase-summary.json` contains all 20 cases for both complete learned
spaces. Each candidate record covers 0–5000 us with 25,001 samples. All candidate
traces were saved before reference arrays were loaded. The reproducible tool is
`tools/diagnose-enriched-linear-matrix.py`; hashes of retained local numerical
arrays and per-record receipts are in `linear-phase-retained-files.json`.

| Candidate | Vector waveform passes | All component/band passes | Maximum relative error |
| --- | ---: | ---: | ---: |
| 264 modes | 19/20 | 11/20 | 35.4677% |
| 392 modes | 19/20 | 11/20 | 40.1145% |

The worst check is the z drive band at receiver index 2 in chirp-20. Increasing
the trained space alone does not resolve it. The existing independent reference
truncation change for that case is 28.2652%; rectangular-05 also has reference
mesh change 21.8376% and truncation change 7.9569%. Those reference failures are
retained in `v12-complete-matrix-audit/audit/`. They prevent treating the current
linear comparison as a converged ground truth for further waveform fitting.

These linear-only diagnostic records and the short native admissions do not
replace the pending complete nonlinear predictions or new converged full 5 ms
nonlinear COMSOL references. No reference waveform was used to fit this basis.

## Shared COMSOL service

The signed read-only research health check still reports another task's source
`d36366154f8950e745a5bd35a6464172f4c96f21e67ecec555045274df614b23`.
The recovered `Research.Common.ps1` protocol checks a single global template
and rejects submissions while another job is active or unresolved. Independent
job directories do not make that service support concurrent submissions. No
remote source was replaced and no new remote job was submitted during this
continuation. Parallel COMSOL work needs a separately configured service and
available compute/license capacity, or a coordinated exclusive time window.
