[CmdletBinding()]
param(
 [Parameter(Mandatory=$true)][string]$Root,
 [Parameter(Mandatory=$true)][ValidatePattern('^comsol-[a-f0-9]{32}$')][string]$JobId
)
$ErrorActionPreference='Stop'
. (Join-Path $Root 'WorkAI.Client.Common.ps1')
$connection=Read-WorkAIConnection (Join-Path $Root 'connection/server-paired-20261022.json')
$key=Join-Path $Root 'private/client-private.xml'
$result=Invoke-WorkAIRequest $connection 'GET' ('/v1/jobs/'+$JobId) ([byte[]]@()) $key
if(-not $result.server_signature_verified){throw 'Unverified server response.'}
if($result.http_status -notin @(200,404)){throw ('Unexpected signed status: '+$result.http_status)}
if($result.http_status -eq 200 -and $result.body.job_id -cne $JobId){throw 'Job identity differs.'}
$result | ConvertTo-Json -Depth 30
