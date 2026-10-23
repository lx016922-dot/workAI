[CmdletBinding()]
param(
    [ValidateSet('validate','submit')][string]$Action='validate',
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$CaseFile,
    [string]$JobId='',
    [ValidateSet('plate_smoke','plate_reference')][string]$Kind='plate_smoke'
)
$ErrorActionPreference='Stop'
try {
    . (Join-Path $Root 'WorkAI.Client.Common.ps1')
    $payload=Join-Path $Root 'volume-protocol-update-20261014-v2/volume-protocol-payload'
    $pins=@{
        'Plate.Common.ps1'='034976f2c70934c0e4b083aeb16ff8d95b6513dd73b7801961fcc81d4506880b'
        'Volume.Protocol.ps1'='b6794d76206b20b18e87de66d3ece499fbcd3b2634ae23ed39f2438be37da75c'
    }
    foreach($name in $pins.Keys){
        if((Get-FileHashText (Join-Path $payload $name)) -cne $pins[$name]){throw 'Original model validator checksum changed.'}
    }
    . (Join-Path $payload 'Plate.Common.ps1')
    $caseJson=[IO.File]::ReadAllText($CaseFile)
    if($caseJson.Length -gt 30000){throw 'Model case is too large.'}
    $case=ConvertFrom-WorkAIJson $caseJson
    try { Assert-PlateCase $case } catch {
        @{valid=$false;validation_error=$_.Exception.Message} | ConvertTo-Json -Compress
        exit 0
    }
    $caseHash=Get-BytesHash $script:Utf8.GetBytes($caseJson)
    if($Action -ceq 'validate'){
        @{valid=$true;case_sha256=$caseHash;schema=$case.schema} | ConvertTo-Json -Compress
        exit 0
    }
    Assert-JobId $JobId
    $connection=Read-WorkAIConnection (Join-Path $Root 'connection/server-paired-20261022.json')
    $key=Join-Path $Root 'private/client-private.xml'
    $health=Invoke-WorkAIRequest $connection 'GET' '/v1/plate/health' ([byte[]]@()) $key
    if($health.http_status -ne 200 -or $health.body.schema -cne 'workai.comsol.plate.health.v1' -or
        $health.body.source_sha256 -cne $connection.plate_source_sha256){throw 'Installed plate model differs from the recovered source pin.'}
    if($case.schema -ceq 'workai.physics_ai.volume.case.v1'){
        $volume=Invoke-WorkAIRequest $connection 'GET' '/v1/volume/health' ([byte[]]@()) $key
        if($volume.http_status -ne 200 -or $volume.body.schema -cne 'workai.comsol.volume.health.v1' -or $volume.body.model_ready -ne $true){throw 'Compatible volume model is not ready.'}
    }
    $directory=Join-Path (Join-Path $Root 'requests') $JobId
    if(Test-Path -LiteralPath $directory){throw 'A request with this ID already exists. Query its status; do not resubmit.'}
    $submission=[ordered]@{job_id=$JobId;kind=$Kind;prepared_at_utc=[DateTime]::UtcNow.ToString('o');case_sha256=$caseHash;source_sha256=$health.body.source_sha256;case_json=$caseJson}
    $null=[IO.Directory]::CreateDirectory($directory)
    Write-JsonFile (Join-Path $directory 'submission.json') $submission
    $body=$script:Utf8.GetBytes(($submission | ConvertTo-Json -Depth 30 -Compress))
    $response=Invoke-WorkAIRequest $connection 'POST' '/v1/plate/jobs' $body $key
    $response | ConvertTo-Json -Depth 30
    if($response.http_status -lt 200 -or $response.http_status -ge 300){exit 2}
} catch {
    [Console]::Error.WriteLine('Model operation failed: '+$_.Exception.Message)
    exit 1
}
