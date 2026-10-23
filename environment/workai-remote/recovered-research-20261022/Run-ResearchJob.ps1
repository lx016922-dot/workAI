#Requires -Version 5.1
param([Parameter(Mandatory=$true)][string]$Root,[Parameter(Mandatory=$true)][string]$JobId)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'Http.Common.ps1')
. (Join-Path $PSScriptRoot 'Research.Common.ps1')
Assert-JobId $JobId
$job=Join-Path (Join-Path $Root 'jobs') $JobId
$started=[DateTime]::UtcNow.ToString('o')
try{
    $request=Read-JsonFile (Join-Path $job 'request.json')
    if((Get-FileHashText (Join-Path $job 'UltraResearch.java')) -cne $request.source_sha256){throw 'Copied research model differs.'}
    Assert-ResearchParameters (ConvertFrom-WorkAIJson $request.parameters_json)
    $settings=Read-JsonFile (Join-Path $Root 'settings.json')
    Write-JsonFile (Join-Path $job 'status.json') @{job_id=$JobId;kind='research_reference';phase='running';worker_pid=$PID;started_at_utc=$started;updated_at_utc=$started}
    Push-Location -LiteralPath $job
    try{& (Join-Path $job 'run-research.ps1') -ComsolBin $settings.comsol_bin *> (Join-Path $job 'worker.log')}finally{Pop-Location}
    if(-not(Test-Path -LiteralPath (Join-Path $job 'completed.properties'))){throw 'No COMSOL completion receipt.'}
    Write-JsonFile (Join-Path $job 'status.json') @{job_id=$JobId;kind='research_reference';phase='completed';worker_pid=$PID;started_at_utc=$started;updated_at_utc=[DateTime]::UtcNow.ToString('o')}
}catch{
    Write-JsonFile (Join-Path $job 'status.json') @{job_id=$JobId;kind='research_reference';phase='failed';worker_pid=$PID;started_at_utc=$started;updated_at_utc=[DateTime]::UtcNow.ToString('o');error=$_.Exception.Message}
    exit 1
}
