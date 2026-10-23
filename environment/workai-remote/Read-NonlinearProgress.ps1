param([Parameter(Mandatory=$true)][string]$JobId)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'WorkAI.Client.Common.ps1')
Assert-JobId $JobId
$c=Read-WorkAIConnection (Join-Path $PSScriptRoot 'connection/server-paired-20261022.json')
$key=Join-Path $PSScriptRoot 'private/client-private.xml'
$r=Invoke-WorkAIRequest $c 'GET' ('/v1/jobs/'+$JobId+'/logs/solver.log') ([byte[]]@()) $key
if($r.http_status -ne 200 -or $r.body.job_id -cne $JobId -or $r.body.name -cne 'solver.log'){throw 'Live log mismatch.'}
$tail=''
if($r.body.tail_base64){$tail=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($r.body.tail_base64));if($tail.Length -gt 2800){$tail=$tail.Substring($tail.Length-2800)}}
@{verified=$r.server_signature_verified;phase=$r.body.phase;sampled_at_utc=$r.body.sampled_at_utc;total_bytes=$r.body.total_bytes;tail=$tail} | ConvertTo-Json
