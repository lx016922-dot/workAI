$ErrorActionPreference='Stop'
$root='/workspace/workai-remote'
. (Join-Path $root 'WorkAI.Client.Common.ps1')
$connection=Read-WorkAIConnection (Join-Path $root 'connection/server-paired-20261022.json')
$reply=Invoke-WorkAIRequest $connection 'GET' '/v1/research/health' ([byte[]]@()) (Join-Path $root 'private/client-private.xml')
$output=[ordered]@{checked_at_utc=[DateTime]::UtcNow.ToString('o');response=$reply}
$output | ConvertTo-Json -Depth 20
