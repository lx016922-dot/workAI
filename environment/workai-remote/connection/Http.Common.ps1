#Requires -Version 5.1
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
$script:Utf8 = New-Object Text.UTF8Encoding($false)
$script:AllowedArtifacts = @('request.json','case.properties','completed.properties','coarse.csv','fine.csv','coarse.mph','fine.mph','compile.log','solver.log','worker.log','status.json')
$script:AllowedArtifacts += @('timecheck.csv','timecheck.mph','coarse.field.csv','fine.field.csv','timecheck.field.csv')

function Write-JsonFile($Path,$Value) {
    $temporary = $Path + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    [IO.File]::WriteAllText($temporary,($Value | ConvertTo-Json -Depth 20),$script:Utf8)
    if ([IO.File]::Exists($Path)) {
        # Windows PowerShell coerces a null string argument to empty; use a real backup path.
        $backup = $temporary + '.previous'
        [IO.File]::Replace($temporary,$Path,$backup)
        [IO.File]::Delete($backup)
    }
    else { [IO.File]::Move($temporary,$Path) }
}
function ConvertFrom-WorkAIJson([string]$Text) {
    # PS 5.1 preserves ISO strings; newer PowerShell needs this explicit option.
    if ((Get-Command ConvertFrom-Json).Parameters.ContainsKey('DateKind')) {
        return (ConvertFrom-Json -InputObject $Text -DateKind String)
    }
    return (ConvertFrom-Json -InputObject $Text)
}
function Read-JsonFile($Path) { return (ConvertFrom-WorkAIJson (Get-Content -Raw -LiteralPath $Path)) }
function Get-BytesHash([byte[]]$Bytes) {
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hash.ComputeHash($Bytes))).Replace('-','').ToLowerInvariant() }
    finally { $hash.Dispose() }
}
function Get-FileHashText($Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Get-UnixTime { return [long]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()) }
function New-RsaFromXml([string]$Xml) {
    $rsa = [Security.Cryptography.RSA]::Create()
    $rsa.FromXmlString($Xml)
    if ($rsa.KeySize -lt 2048) { $rsa.Dispose(); throw 'RSA key must be at least 2048 bits.' }
    return $rsa
}
function Get-RequestText($Method,$Path,$Timestamp,$Nonce,$BodyHash) {
    return "workai-comsol-http-v1`n$Method`n$Path`n$Timestamp`n$Nonce`n$BodyHash"
}
function Get-ResponseText($Status,$Nonce,$BodyHash) {
    return "workai-comsol-http-response-v1`n$Status`n$Nonce`n$BodyHash"
}
function Get-Signature($Rsa,[string]$Text) {
    return [Convert]::ToBase64String($Rsa.SignData($script:Utf8.GetBytes($Text),[Security.Cryptography.HashAlgorithmName]::SHA256,[Security.Cryptography.RSASignaturePadding]::Pkcs1))
}
function Test-RequestSignature($Rsa,$Method,$Path,$Timestamp,$Nonce,$Signature,[byte[]]$Body,$Seen) {
    if ($Timestamp -notmatch '^\d{10,11}$' -or $Nonce -notmatch '^[a-f0-9]{32}$') { return $false }
    $now = Get-UnixTime
    if ([Math]::Abs($now-[long]$Timestamp) -gt 300 -or $Seen.ContainsKey($Nonce)) { return $false }
    try {
        $canonical = Get-RequestText $Method $Path $Timestamp $Nonce (Get-BytesHash $Body)
        $valid = $Rsa.VerifyData($script:Utf8.GetBytes($canonical),[Convert]::FromBase64String($Signature),[Security.Cryptography.HashAlgorithmName]::SHA256,[Security.Cryptography.RSASignaturePadding]::Pkcs1)
    } catch { return $false }
    if (-not $valid) { return $false }
    foreach ($key in @($Seen.Keys)) { if ($Seen[$key] -lt $now-300) { $Seen.Remove($key) } }
    if ($Seen.Count -ge 10000) { return $false }
    $Seen[$Nonce] = $now
    return $true
}
function Assert-JobId([string]$Id) {
    if ($Id -cnotmatch '^comsol-[a-f0-9]{32}$') { throw 'Invalid job ID.' }
}
function Assert-Template($TemplateRoot,$Manifest) {
    foreach ($item in $Manifest.files.PSObject.Properties) {
        if ($item.Name -notin @('request.json','case.properties','UltraBenchmark.java','run.ps1','Run-Job.ps1','Http.Common.ps1','Serve-Comsol.ps1')) { throw 'Unexpected bundle manifest entry.' }
        $path = if ($item.Name -in @('Run-Job.ps1','Http.Common.ps1','Serve-Comsol.ps1')) { Join-Path (Split-Path $TemplateRoot -Parent) $item.Name } else { Join-Path $TemplateRoot $item.Name }
        if ((Get-FileHashText $path) -cne $item.Value) { throw "Installed template changed: $($item.Name). Rebuild and review the bundle." }
    }
    if (@($Manifest.files.PSObject.Properties).Count -ne 7) { throw 'Incomplete bundle manifest.' }
}
function New-JobFiles($Root,$Submission) {
    $expected = @('job_id','kind','prepared_at_utc','case_sha256','source_sha256')
    $actual = @($Submission.PSObject.Properties.Name)
    if ($actual.Count -ne $expected.Count -or @($actual | Where-Object { $_ -notin $expected }).Count) { throw 'Unexpected job fields.' }
    Assert-JobId $Submission.job_id
    if ($Submission.kind -cnotin @('smoke','benchmark')) { throw 'Only smoke and benchmark jobs are supported.' }
    if ($Submission.prepared_at_utc -notmatch '(Z|\+00:00)$') { throw 'UTC preparation time required.' }
    $prepared = [DateTimeOffset]::Parse($Submission.prepared_at_utc,[Globalization.CultureInfo]::InvariantCulture)
    if ($prepared -gt [DateTimeOffset]::UtcNow.AddMinutes(5) -or $prepared -lt [DateTimeOffset]::UtcNow.AddDays(-1)) { throw 'Prepare a fresh job before submission.' }
    $template = Join-Path $Root 'template'
    $manifest = Read-JsonFile (Join-Path $Root 'bundle-manifest.json')
    Assert-Template $template $manifest
    $request = Read-JsonFile (Join-Path $template 'request.json')
    if ($Submission.case_sha256 -cne $request.case_sha256 -or $Submission.source_sha256 -cne $request.source_sha256) { throw 'This endpoint only runs the reviewed, bundled model and case.' }
    $jobs = Join-Path $Root 'jobs'
    $job = Join-Path $jobs $Submission.job_id
    if (Test-Path -LiteralPath $job) {
        $old = Read-JsonFile (Join-Path $job 'submission.json')
        foreach ($key in $expected) { if ($old.$key -cne $Submission.$key) { throw 'Job ID already belongs to a different request.' } }
        return [pscustomobject]@{ Directory=$job; Created=$false }
    }
    foreach ($directory in @(Get-ChildItem -LiteralPath $jobs -Directory)) {
        $statePath = Join-Path $directory.FullName 'status.json'
        if (-not (Test-Path $statePath)) { throw 'An incomplete job exists; inspect it locally before submitting more work.' }
        $state = Read-JsonFile $statePath
        if ($state.phase -in @('starting','running')) { throw 'A job is already active. Inspect its status; do not resubmit.' }
    }
    $null = New-Item -ItemType Directory -Path $job
    Write-JsonFile (Join-Path $job 'submission.json') $Submission
    $request.job_id = $Submission.job_id
    $request.prepared_at_utc = $Submission.prepared_at_utc
    if ($Submission.kind -eq 'smoke') {
        $request.schema = 'workai.comsol.smoke.v1'
        $request.scope = 'license_and_api_smoke_only_not_an_ascan_benchmark'
        $request.coarse_elements = 8
        $request.fine_elements = 16
        $request.positive_frequency_count = 2
        $request.frequency_step_hz = [double]$request.case.center_mhz * 1000000
    }
    Write-JsonFile (Join-Path $job 'request.json') $request
    $props = [IO.File]::ReadAllText((Join-Path $template 'case.properties'))
    foreach ($name in @('job_id','coarse_elements','frequency_step_hz','positive_frequency_count')) {
        $value = [Convert]::ToString($request.$name,[Globalization.CultureInfo]::InvariantCulture)
        $props = [Regex]::Replace($props,('(?m)^'+$name+'=[^\r\n]*'),($name+'='+$value))
    }
    [IO.File]::WriteAllText((Join-Path $job 'case.properties'),$props,$script:Utf8)
    foreach ($name in @('UltraBenchmark.java','run.ps1')) { Copy-Item -LiteralPath (Join-Path $template $name) -Destination (Join-Path $job $name) }
    Write-JsonFile (Join-Path $job 'status.json') ([ordered]@{ job_id=$Submission.job_id; kind=$Submission.kind; phase='starting'; updated_at_utc=[DateTime]::UtcNow.ToString('o') })
    return [pscustomobject]@{ Directory=$job; Created=$true }
}
function Get-JobStatus($Root,$Id) {
    Assert-JobId $Id
    $job = Join-Path (Join-Path $Root 'jobs') $Id
    $state = Read-JsonFile (Join-Path $job 'status.json')
    $artifacts = @()
    if ($state.phase -in @('completed','failed')) {
        foreach ($name in $script:AllowedArtifacts) {
            $path = Join-Path $job $name
            if (Test-Path -LiteralPath $path -PathType Leaf) {
                $artifacts += [pscustomobject]@{ name=$name; bytes=(Get-Item -LiteralPath $path).Length; sha256=(Get-FileHashText $path) }
            }
        }
    }
    return [ordered]@{ schema='workai.comsol.http.job_status.v1'; job_id=$Id; kind=$state.kind; phase=$state.phase; state=$state; artifacts=$artifacts }
}
