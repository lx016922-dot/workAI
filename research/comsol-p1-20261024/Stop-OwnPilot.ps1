# Stop only the identified P1 pilot and its descendants. Keep the HTTP API open.
# Run on the Windows COMSOL server, from an administrator PowerShell.
[CmdletBinding(SupportsShouldProcess=$true,ConfirmImpact='Medium')]
param(
 [string]$Root='C:\Users\Administrator\Desktop\workai-comsol-http-v1\workai-comsol-http-v1',
 [ValidatePattern('^comsol-[a-f0-9]{32}$')][string]$JobId='comsol-553288731cd144e18544c0106f381838'
)
$ErrorActionPreference='Stop'
$job=Join-Path (Join-Path $Root 'jobs') $JobId
$statusPath=Join-Path $job 'status.json'
$state=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
if($state.job_id -cne $JobId){throw 'Job identity differs; no process stopped.'}
if($state.phase -in @('completed','failed')){Write-Output 'Job already finished; no process stopped.';return}
$workerId=[int]$state.worker_pid
$processes=@(Get-CimInstance Win32_Process)
$worker=@($processes | Where-Object {$_.ProcessId -eq $workerId})
if($worker.Count -ne 1 -or $worker[0].CommandLine -notlike ('*'+$JobId+'*') -or $worker[0].CommandLine -notlike '*Run-ResearchJob.ps1*'){
 throw 'Worker PID/command line not verified. No process stopped and no status changed.'
}
$targets=New-Object 'System.Collections.Generic.List[int]'
$targets.Add($workerId)
for($i=0;$i -lt $targets.Count;$i++){
 foreach($child in @($processes | Where-Object {$_.ParentProcessId -eq $targets[$i]})){
  if(-not $targets.Contains([int]$child.ProcessId)){$targets.Add([int]$child.ProcessId)}
 }
}
if(-not $PSCmdlet.ShouldProcess(($JobId+'; process IDs '+($targets -join ',')),'Stop this pilot and preserve all files')){return}
for($i=$targets.Count-1;$i -ge 0;$i--){
 $current=Get-CimInstance Win32_Process -Filter ('ProcessId='+$targets[$i])
 $original=@($processes | Where-Object {$_.ProcessId -eq $targets[$i]})
 if($null -ne $current -and $original.Count -eq 1){
  if($current.CreationDate -ne $original[0].CreationDate){throw 'PID was reused; stopped without touching that process.'}
  Stop-Process -Id $targets[$i] -Force -ErrorAction SilentlyContinue
 }
}
$state.phase='failed'
$state | Add-Member -Force -NotePropertyName error -NotePropertyValue 'Pilot stopped by operator to update per-time-step Jacobian settings; partial output is not a delivered result.'
$state.updated_at_utc=[DateTime]::UtcNow.ToString('o')
$temporary=$statusPath+'.cancel.tmp'
[IO.File]::WriteAllText($temporary,($state | ConvertTo-Json -Depth 20),[Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $temporary -Destination $statusPath -Force
Write-Output ('Stopped only '+$JobId+'. Existing result files and the HTTP API were preserved.')
