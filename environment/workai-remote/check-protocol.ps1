$ErrorActionPreference = 'Stop'
$root = '/workspace/workai-remote/volume-protocol-update-20261014-v2'
$files = @(Get-ChildItem -LiteralPath $root -Filter '*.ps1' -Recurse)
if ($files.Count -ne 4) { throw 'Expected all four PowerShell files from the uploaded update.' }
foreach ($file in $files) {
    $tokens = $null; $errors = $null
    $null = [Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw ($file.Name + ': ' + ($errors.Message -join ' | ')) }
}
Write-Output 'PASS: all four original PowerShell files parse under PowerShell 7.6.7.'

# These files define validation functions only. Do not execute the updater,
# API listener, model compilation, job submission, or remote requests.
. (Join-Path $root 'volume-protocol-payload/Plate.Common.ps1')
$case = @'
{
  "schema": "workai.physics_ai.volume.case.v1",
  "geometry": {"length_mm": 200, "width_mm": 100, "thickness_mm": 2},
  "material": {"rho_kg_m3": 2700, "cl_m_s": 6320, "ct_m_s": 3130},
  "boundary": {"x_min": "absorbing", "x_max": "absorbing", "y_min": "free", "y_max": "free", "z_min": "free", "z_max": "free"},
  "notch": {"enabled": false, "x_mm": 100, "y_mm": 50, "length_mm": 5, "width_mm": 0.5, "depth_mm": 1},
  "sources": [{"id": "tx0", "kind": "prescribed_surface_displacement", "position_mm": [10,50,2], "direction": [0,0,1], "width_mm": 1, "span_mm": 1, "amplitude_nm": 1, "frequencies_mhz": [0.2], "cycles": 5, "delay_us": 0, "phase_deg": 0}],
  "receivers": [{"id": "rx0", "position_mm": [100,50,1]}],
  "acquisition": {"duration_us": 150, "sample_step_us": 0.5}
}
'@ | ConvertFrom-Json
Assert-PlateCase $case
Write-Output 'PASS: original validator accepts a 200 x 100 x 2 mm displacement case with an internal receiver.'
$case.sources[0].position_mm = @(10,50,3)
$rejected = $false
try { Assert-PlateCase $case } catch { $rejected = $true }
if (-not $rejected) { throw 'The original validator accepted a source outside the plate.' }
Write-Output 'PASS: original validator rejects a source outside the plate.'
Write-Output 'NOT TESTED: Windows runtime, remote authentication, COMSOL license, model execution, or convergence.'
