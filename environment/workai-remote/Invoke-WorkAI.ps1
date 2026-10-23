[CmdletBinding()]
param(
    [ValidateSet('probe','health','volume-health','plate-health','smoke','job')][string]$Action='health',
    [string]$JobId='',
    [string]$ConnectionFile=(Join-Path $PSScriptRoot 'connection/server-paired-20261022.json')
)
$ErrorActionPreference = 'Stop'
try {
    . (Join-Path $PSScriptRoot 'WorkAI.Client.Common.ps1')
    $connection = Read-WorkAIConnection $ConnectionFile
    $method = 'GET'; $path = '/v1/health'; $body = [byte[]]@()
    switch ($Action) {
        'volume-health' { $path = '/v1/volume/health' }
        'plate-health' { $path = '/v1/plate/health' }
        'job' { Assert-JobId $JobId; $path = '/v1/jobs/'+$JobId }
        'smoke' {
            # Require an explicit stable ID, retain the exact request before
            # sending, and never retry POST automatically after a lost reply.
            Assert-JobId $JobId
            $method = 'POST'; $path = '/v1/jobs'
            $directory = Join-Path (Join-Path $PSScriptRoot 'requests') $JobId
            $requestFile = Join-Path $directory 'submission.json'
            if (Test-Path -LiteralPath $requestFile) { $submission = Read-JsonFile $requestFile }
            else {
                $submission = [ordered]@{job_id=$JobId;kind='smoke';prepared_at_utc=[DateTime]::UtcNow.ToString('o');case_sha256=$connection.case_sha256;source_sha256=$connection.source_sha256}
                $null = [IO.Directory]::CreateDirectory($directory)
                Write-JsonFile $requestFile $submission
            }
            if ($submission.job_id -cne $JobId -or $submission.kind -cne 'smoke') { throw 'Stored job request differs.' }
            $body = $script:Utf8.GetBytes(($submission | ConvertTo-Json -Depth 20 -Compress))
            [Console]::Error.WriteLine('Retained request for '+$JobId+'. If a response is lost, inspect this ID before submitting again.')
        }
    }
    $result = Invoke-WorkAIRequest $connection $method $path $body (Join-Path $PSScriptRoot 'private/client-private.xml') -Unsigned:($Action -eq 'probe')
    $result | ConvertTo-Json -Depth 20
    if ($Action -eq 'probe') {
        if ($result.http_status -ne 401) { throw 'Probe expected a signed authentication-required response.' }
        exit 0
    }
    if ($result.http_status -lt 200 -or $result.http_status -ge 300) { exit 2 }
    $expectedSchema = switch ($Action) {
        'health' {'workai.comsol.http.health.v1'}
        'plate-health' {'workai.comsol.plate.health.v1'}
        'volume-health' {'workai.comsol.volume.health.v1'}
        default {'workai.comsol.http.job_status.v1'}
    }
    if ($result.body.schema -cne $expectedSchema) { throw 'Unexpected response schema.' }
} catch {
    [Console]::Error.WriteLine('WorkAI request failed: '+$_.Exception.Message)
    exit 1
}
