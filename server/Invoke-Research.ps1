[CmdletBinding()]
param(
    [ValidateSet('health','validate','submit')][string]$Action='health',
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$SceneFile='', [string]$ParametersFile='', [string]$JobId=''
)
$ErrorActionPreference='Stop'
try {
    . (Join-Path $Root 'WorkAI.Client.Common.ps1')
    $recovered=Join-Path $Root 'recovered-research-20261022'
    $sourceHash='ac090340bd65c47240b7954d2a51125b010f18d0e608bed012a6525febacccd7'
    $pins=@{
        'Research.Common.ps1'='4a20b13f3410cfc37935860a653deca6d4c15cf23fdad109123dc400f7fa8748'
        'research-template/UltraResearch.java'=$sourceHash
    }
    foreach($name in $pins.Keys){if((Get-FileHashText (Join-Path $recovered $name)) -cne $pins[$name]){throw 'Recovered research input checksum changed.'}}
    . (Join-Path $recovered 'Research.Common.ps1')
    $modelManifest=Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'comsol/model.json') | ConvertFrom-Json
    $newSourceHash=[string]$modelManifest.source_sha256
    if($modelManifest.schema -cne 'workai.comsol.research.revision.v1' -or $newSourceHash -cnotmatch '^[a-f0-9]{64}$' -or (Get-FileHashText (Join-Path $PSScriptRoot 'comsol/UltraResearch.java')) -cne $newSourceHash){throw 'Reviewed model source checksum changed.'}
    if($Action -cne 'health'){
        $sceneText=[IO.File]::ReadAllText($SceneFile)
        $parameterText=[IO.File]::ReadAllText($ParametersFile)
        if($sceneText.Length -gt 15000 -or $parameterText.Length -gt 15000){throw 'Research input exceeds supported size.'}
        try { Assert-ResearchParameters (ConvertFrom-WorkAIJson $parameterText) } catch {
            if($Action -ceq 'validate'){@{valid=$false;validation_error=$_.Exception.Message} | ConvertTo-Json -Compress;exit 0}
            throw
        }
        if($Action -ceq 'validate'){
            @{valid=$true;case_sha256=(Get-BytesHash $script:Utf8.GetBytes($sceneText));parameters_sha256=(Get-BytesHash $script:Utf8.GetBytes($parameterText))} | ConvertTo-Json -Compress
            exit 0
        }
    }
    $connection=Read-WorkAIConnection (Join-Path $Root 'connection/server-paired-20261022.json')
    $key=Join-Path $Root 'private/client-private.xml'
    $health=Invoke-WorkAIRequest $connection 'GET' '/v1/research/health' ([byte[]]@()) $key
    if($Action -ceq 'health'){$health | ConvertTo-Json -Depth 30;exit 0}
    if($health.http_status -ne 200 -or $health.body.schema -cne 'workai.comsol.research.health.v1' -or $health.body.status -cne 'ready_for_job' -or $health.body.source_sha256 -cnotin @($sourceHash,$newSourceHash)){throw 'Installed research model differs from reviewed source versions.'}
    $scene=ConvertFrom-WorkAIJson $sceneText
    if($scene.schema -ceq $modelManifest.nonlinear_scene -and $health.body.source_sha256 -cne $newSourceHash){throw 'Nonlinear model revision is not installed.'}
    $sourceHash=[string]$health.body.source_sha256
    Assert-JobId $JobId
    $directory=Join-Path (Join-Path $Root 'requests') $JobId
    if(Test-Path -LiteralPath $directory){throw 'Request ID already exists. Query it; do not resubmit.'}
    $submission=[ordered]@{job_id=$JobId;kind='research_reference';prepared_at_utc=[DateTime]::UtcNow.ToString('o');scene_sha256=(Get-BytesHash $script:Utf8.GetBytes($sceneText));source_sha256=$sourceHash;parameters_sha256=(Get-BytesHash $script:Utf8.GetBytes($parameterText));scene_json=$sceneText;parameters_json=$parameterText}
    $null=[IO.Directory]::CreateDirectory($directory)
    Write-JsonFile (Join-Path $directory 'submission.json') $submission
    $body=$script:Utf8.GetBytes(($submission | ConvertTo-Json -Depth 30 -Compress))
    $response=Invoke-WorkAIRequest $connection 'POST' '/v1/research/jobs' $body $key
    $response | ConvertTo-Json -Depth 30
    if($response.http_status -ne 202){exit 2}
} catch {
    [Console]::Error.WriteLine('Research operation failed: '+$_.Exception.Message)
    exit 1
}
