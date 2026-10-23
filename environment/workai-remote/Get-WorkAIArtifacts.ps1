[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$JobId,
    [string[]]$Names=@('request.json','completed.properties','coarse.csv','fine.csv','solver.log','worker.log','coarse.mph','fine.mph'),
    [string]$ConnectionFile=(Join-Path $PSScriptRoot 'connection/server-paired-20261022.json')
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'WorkAI.Client.Common.ps1')
Assert-JobId $JobId
$connection = Read-WorkAIConnection $ConnectionFile
$privateKey = Join-Path $PSScriptRoot 'private/client-private.xml'
$status = Invoke-WorkAIRequest $connection 'GET' ('/v1/jobs/'+$JobId) ([byte[]]@()) $privateKey
if ($status.http_status -ne 200 -or $status.body.schema -cne 'workai.comsol.http.job_status.v1' -or
    $status.body.job_id -cne $JobId -or $status.body.phase -cnotin @('completed','failed')) {
    throw 'Expected the signed final status for this exact job before downloading artifacts.'
}
$targets = @()
foreach ($name in $Names) {
    if ($name -cnotin $script:AllowedArtifacts) { throw ('Artifact is not allowlisted: '+$name) }
    $entry = @($status.body.artifacts | Where-Object { $_.name -ceq $name })
    if ($entry.Count -ne 1) { throw ('Artifact absent or ambiguous in signed job status: '+$name) }
    if ($entry[0].bytes -lt 0 -or $entry[0].bytes -gt 1048576 -or $entry[0].sha256 -cnotmatch '^[a-f0-9]{64}$') {
        throw 'This helper supports verified artifacts up to 1 MiB each; larger results need a streaming downloader.'
    }
    $targets += $entry[0]
}
$directory = Join-Path (Join-Path (Join-Path $PSScriptRoot 'requests') $JobId) 'artifacts'
$null = [IO.Directory]::CreateDirectory($directory)
$verified = @()
foreach ($entry in $targets) {
    $destination = Join-Path $directory $entry.name
    if (Test-Path -LiteralPath $destination) {
        $existing = Get-Item -LiteralPath $destination -Force
        if ($existing -isnot [IO.FileInfo] -or ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            $existing.Length -ne $entry.bytes -or (Get-FileHashText $destination) -cne $entry.sha256) {
            throw ('Preserving an existing unverified artifact: '+$destination)
        }
    } else {
        $reply = Invoke-WorkAIRequest $connection 'GET' ('/v1/jobs/'+$JobId+'/files/'+$entry.name) ([byte[]]@()) $privateKey -ReturnBytes
        if ($reply.http_status -ne 200 -or $reply.body_bytes.Length -ne $entry.bytes -or
            (Get-BytesHash $reply.body_bytes) -cne $entry.sha256) {
            throw ('Downloaded bytes differ from the signed artifact manifest: '+$entry.name)
        }
        $stream = [IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try { $stream.Write($reply.body_bytes,0,$reply.body_bytes.Length) } finally { $stream.Dispose() }
    }
    $verified += [pscustomobject]@{name=$entry.name;bytes=$entry.bytes;sha256=$entry.sha256;path=$destination}
}
[pscustomobject]@{job_id=$JobId;phase=$status.body.phase;server_signature_verified=$true;artifacts=$verified} | ConvertTo-Json -Depth 5
