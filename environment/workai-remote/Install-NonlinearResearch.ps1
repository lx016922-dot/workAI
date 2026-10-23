[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$SourceFile,[Parameter(Mandatory=$true)][string]$ExpectedSourceSha256)
$ErrorActionPreference='Stop'
$workaiRoot=$PSScriptRoot
. (Join-Path $workaiRoot 'WorkAI.Client.Common.ps1')
$connection=Read-WorkAIConnection (Join-Path $workaiRoot 'connection/server-paired-20261022.json')
$key=Join-Path $workaiRoot 'private/client-private.xml'
$source=[IO.File]::ReadAllText($SourceFile)
$sourceHash=Get-BytesHash $script:Utf8.GetBytes($source)
$health=Invoke-WorkAIRequest $connection 'GET' '/v1/research/health' ([byte[]]@()) $key
if($health.http_status -ne 200 -or $health.body.status -cne 'ready_for_job' -or $health.body.source_sha256 -cne $ExpectedSourceSha256){throw 'Installed source differs from expected revision; no update sent.'}
$revision='research-'+[Guid]::NewGuid().ToString('N')
$directory=Join-Path (Join-Path $workaiRoot 'model-revisions') $revision
$null=[IO.Directory]::CreateDirectory($directory)
$submission=[ordered]@{revision_id=$revision;expected_source_sha256=$ExpectedSourceSha256;source_sha256=$sourceHash;java_source=$source}
Write-JsonFile (Join-Path $directory 'submission.json') $submission
$body=$script:Utf8.GetBytes(($submission | ConvertTo-Json -Depth 5 -Compress))
if($body.Length -gt 65536){throw 'Model update exceeds signed request limit.'}
@{revision_id=$revision;source_sha256=$sourceHash;request_bytes=$body.Length;attempted_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress
try{
    $result=Invoke-WorkAIRequest $connection 'POST' '/v1/research/revisions' $body $key
    Write-JsonFile (Join-Path $directory 'response.json') $result
    $result | ConvertTo-Json -Depth 20
}catch{
    Write-JsonFile (Join-Path $directory 'uncertain.json') @{error=$_.Exception.Message;message='Do not repeat POST. Check signed research health for exact source hash.'}
    throw
}
