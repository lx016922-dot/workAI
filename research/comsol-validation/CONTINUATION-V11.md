# V11 continuation: nonlinear modal graph research

This is a continuation of the software published at [v0.3.0](https://github.com/lx016922-dot/workAI/releases/tag/v0.3.0).
The default offline weights are unchanged. **V11 has not met the complete 5% accuracy gate.**

The verification matrix remains Hann, Gaussian, rectangular and chirp excitation,
each with 1, 3, 5, 10 and 20 cycles, eight three-component receivers, and complete
5000 microsecond records. Linear, geometric-nonlinear and full Murnaghan controls
start from zero. No time shift, amplitude rescaling or shortened analysis window
is allowed. Nonlinear increments are checked separately from total displacement.

## Research implementation

The new candidate uses its own learned graph tangent to construct a modal basis.
COMSOL arrays do not enter candidate inference. A symmetric fourth-order
composition combines accurately driven candidate linear flow with conservative
nonlinear increments. The external forcing uses DOP853 at a fixed 1e-12 relative
tolerance; free evolution uses the candidate's exact oscillator solution.
Every entry of the learned quadratic/cubic material tensors and
every original quadrature point contributes to the nonlinear force. Truncating the
modal basis still requires independent convergence evidence.

For the rectangular plate under uniform axial traction, exact reflection symmetry
allows a quarter computational domain. Transverse displacement remains present:
its sign changes under the corresponding reflection. The quarter domain fixes only
the appropriate normal displacement at each symmetry plane. Asymmetric loads and
material tensors are rejected. Full-domain/quarter-domain tests independently
check mass, traction, all three constitutive energies and matching eigenfrequencies.

The P6 candidate uses 12 x 3 x 3 quarter-domain elements, with grading ratios
50 x 12 x 12, 26,353 nodes and 200 modes in the loaded symmetry class. Its basis is
compiled from the candidate weights alone. Compiler source snapshots, input and
artifact hashes are retained.

Training retains exact affine/quadratic derivative consistency, summation by parts,
fixed-boundary conditions and objective material energy. Analytic eigenfrequency
sensitivities avoid finite differences of large eigensolves. Each sequential
optimization step rechecks actual eigenfrequencies and the full nonlinear physical
constraints before acceptance.

## Evidence available so far

| Check | Result | Scope |
| --- | --- | --- |
| Moderate graded COMSOL full-domain mesh pair | Maximum relevant frequency change 1.14277 Hz | Fails the 0.5 Hz phase-reference gate |
| Quarter-domain COMSOL, uniform cross section, h = 0.3 / 0.2 mm | Maximum relevant frequency change 6.19566 Hz | Mesh geometry, reflections and modal energy identities verified; phase gate fails |
| Quarter candidate linear development matrix, first 53 modes | 19/20 vector checks; 15/20 complete component/band checks | Actual 5 ms linear responses; not nonlinear certification |
| Quarter candidate versus existing coarse native COMSOL 20 us record | Geometric increment 15.066%; material increment 11.097% maximum vector error | Fails; the coarse reference also needs spatial/time convergence |
| Formal physical training, 576 fields + 288 independent fields | Both pass; validation maximum 0.04662% | Physical residuals are not waveform errors |
| Pinned offline runtime numerical tests | 30 tests pass | Force/energy consistency, sensitivity, mesh, symmetry and independent 5 ms closed-form forcing checks |
| Corrected temporal layer, native 20 us scene, 200 candidate modes | 0.1 versus 0.025 us: maximum nonlinear-increment vector change 0.531%; complete component/band change 3.234% | Short temporal convergence only; not a COMSOL accuracy certificate |
| Candidate versus new native quarter COMSOL, h = 0.75 mm, 20 us | Maximum vector errors: linear 0.530%, geometric increment 7.067%, material increment 5.645%, total nonlinear increment 5.210%; maximum complete check 28.926% | Fails; independent finer native mesh/time checks remain pending |
| Explicit cross-section graded quarter COMSOL, h = 0.3 mm | Rejected: actual y spacing decreases toward the symmetry plane | Mesh orientation differs from the requested outer-surface refinement; excluded from training and convergence admission |

The first formal optimization changes the seven-mode frequency-error norm only
from 2.433461 to 2.433270 Hz. The further run reduces this norm to 2.317378 Hz, with maximum training/independent
physical residuals of 0.82185%. It reduces the soft physical loss weight while
retaining the same 2.5% hard per-constraint limit and fresh independent validation
fields. Neither training result is an accuracy certificate.

An independent forcing diagnostic found that the original external-force kick
scheme alone caused 6.934% vector error for the rectangular five-cycle case at a
0.2 us step (29.729% including all component/band checks). The two original batch
runs, 40201236214 and 40201622534, were cancelled and retained as incomplete
evidence. The corrected implementation evolves nonlinear increments around the
accurately forced candidate linear trajectory. The new full-duration plan is
`plans/v11-quarter-driven.json`, with a 0.1 us internal step. Existing compiled
bases remain immutable: their graph tangent and eigenvectors are unchanged, while
each prediction records the separate corrected runtime source hashes.

Native quarter-domain COMSOL spatial/time refinements and an explicitly graded
quarter-domain reference pair are queued separately. The signed transport keeps
the original time-step, export, polynomial-order and element-budget limits.
Each source revision is compiled through the existing compare-and-swap endpoint;
the server retains its previous source and manifest. Unknown submission outcomes
must be resolved by querying the original job identity, never by duplicate POSTs.

The actual-axis audit detected that COMSOL's y-edge orientation reverses the
requested one-sided grading. `UltraResearchV11QuarterGradedChecked.java` corrects
that edge and checks all actual axis counts, spacing directions, grading ratios
and Cartesian element counts before solving. The original completed result and
its rejection remain archived. The corrected variant has not yet supplied an
admitted reference.

`UltraResearchV11QuarterNodal.java` prepares a separate research reference: native
COMSOL eigenfields at every Q4 tensor-element node. Coordinates and eigenfrequencies
are stored once in the signed receipt; the compact CSV retains all three field
components in double precision. Two prepared exports contain 78 modes at h =
0.35 mm and 56 at h = 0.3 mm, each below the unchanged two-million-record limit.
The 256 MiB decoded-export limit also remains enforced. They wait behind the native refinement queue.
These are inputs to a future independently validated nonlinear modal reference,
not completed nonlinear COMSOL transients.

The current complete nonlinear matrix is run
[40204192440](https://github.com/lx016922-dot/workAI/actions/runs/40204192440),
frozen at commit `1cb35bc82c056b9960f3e94d4f7c679afe8f2ee6`.
`tools/audit-modal-matrix.py` independently reloads its saved arrays, verifies the
complete time grid and model identity, reconstructs receivers from saved modal
coordinates, checks energy histories and sampled conservative potentials, and
regenerates each linear COMSOL comparison. The separate audit/publication workflow
requires all 20 source jobs to finish before running. Its acceptance flag remains
false without full-duration nonlinear reference and convergence evidence.

## Reproduction

Use the Python runtime included in the existing offline distribution, or Python
3.12 with NumPy 2.2.6, SciPy 1.15.3, Numba 0.63.1 and llvmlite 0.46.0. All model
prediction and archived-reference comparisons run without a network connection.
Live COMSOL regeneration requires the separately configured licensed server.

```sh
python tools/run-modal-candidate-case.py \
  --model /path/to/extracted/research-model \
  --plan research/comsol-validation/plans/v11-quarter-driven.json \
  --case hann-01 --step-us 0.1 --modes 200 --output results \
  --reference research/comsol-validation/data/eigen-graded20-030-v11 \
  --check-reference research/comsol-validation/data/eigen-graded20-035-v11
```

The runner saves all three complete predictions before loading reference arrays.
It retains modal coordinates, receiver histories, energy histories, source hashes,
model identity and separate reference-convergence checks. A linear reference cannot
set the nonlinear accuracy flag. The research workflow is
`.github/workflows/comsol-modal-validation.yml`; a successful workflow means the
calculation completed, not that the scientific 5% gate passed.
