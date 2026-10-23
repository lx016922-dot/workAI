$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'WorkAI.Client.Common.ps1')
. (Join-Path $PSScriptRoot 'pairing/Repair-CloudPairing.ps1')
$script:ChecksPassed = 0
function Check([bool]$Condition,[string]$Label) {
    if (-not $Condition) { throw ('CHECK FAILED: '+$Label) }
    $script:ChecksPassed++
    Write-Output ('PASS: '+$Label)
}
function Expect-Rejection([scriptblock]$Operation,[string]$Label) {
    $rejected = $false
    try { $null = & $Operation } catch { $rejected = $true }
    Check $rejected $Label
}

$checkFiles = @('WorkAI.Client.Common.ps1','New-WorkAIIdentity.ps1','Invoke-WorkAI.ps1','pairing/Repair-CloudPairing.ps1','connection/Http.Common.ps1')
foreach ($name in $checkFiles) {
    $tokens = $null; $errors = $null
    $null = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot $name),[ref]$tokens,[ref]$errors)
    if ($errors.Count) { throw ($name+': '+($errors.Message -join ' | ')) }
}
Check $true 'cloud client, identity helper, pairing tool, and uploaded protocol parse'
$connection = Read-WorkAIConnection (Join-Path $PSScriptRoot 'connection/server-original.json')
Check ($connection.base_url -ceq 'http://100.100.205.105:18765') 'uploaded server public-key pin and connection schema verify'
$client = New-RsaFromXml ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'private/client-private.xml')))
$clientPublic = New-RsaFromXml ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'pairing/cloud-client-public.xml')))
$mockServer = [Security.Cryptography.RSA]::Create(3072)
$oldClient = [Security.Cryptography.RSA]::Create(3072)
$fixtureRoot = Join-Path '/tmp' ('workai-pairing-checks-'+[Guid]::NewGuid().ToString('N'))
$null = [IO.Directory]::CreateDirectory($fixtureRoot)
try {
    $body = $script:Utf8.GetBytes('{"test":"signature compatibility"}')
    $headers = New-WorkAIRequestHeaders $client 'POST' '/v1/jobs' $body
    $seen = @{}
    $arguments = @($clientPublic,'POST','/v1/jobs',$headers['X-WorkAI-Date'],$headers['X-WorkAI-Nonce'],$headers['X-WorkAI-Signature'],$body,$seen)
    Check (Test-RequestSignature @arguments) 'original server verifier accepts the new cloud client signature'
    Check (-not (Test-RequestSignature @arguments)) 'original server verifier rejects replayed nonce'
    $changedBody = $script:Utf8.GetBytes('{"test":"changed"}')
    Check (-not (Test-RequestSignature $clientPublic 'POST' '/v1/jobs' $headers['X-WorkAI-Date'] $headers['X-WorkAI-Nonce'] $headers['X-WorkAI-Signature'] $changedBody @{})) 'original server verifier rejects a tampered request body'
    $staleDate = [string]((Get-UnixTime)-601)
    $staleSignature = Get-Signature $client (Get-RequestText 'POST' '/v1/jobs' $staleDate $headers['X-WorkAI-Nonce'] (Get-BytesHash $body))
    Check (-not (Test-RequestSignature $clientPublic 'POST' '/v1/jobs' $staleDate $headers['X-WorkAI-Nonce'] $staleSignature $body @{})) 'original server verifier rejects a stale signed request'

    $replyBody = $script:Utf8.GetBytes('{"status":"fixture-only"}')
    $nonce = [Guid]::NewGuid().ToString('N')
    $hash = Get-BytesHash $replyBody
    $reply = @{'X-WorkAI-Nonce'=$nonce;'X-WorkAI-Body-SHA256'=$hash;'X-WorkAI-Signature'=(Get-Signature $mockServer (Get-ResponseText 200 $nonce $hash))}
    Assert-WorkAIResponse $mockServer 200 $nonce $reply $replyBody
    Check $true 'cloud verifier accepts an original-protocol signed response fixture'
    Expect-Rejection { Assert-WorkAIResponse $mockServer 200 $nonce $reply $changedBody } 'cloud verifier rejects a changed response body'
    Expect-Rejection { Assert-WorkAIResponse $mockServer 200 ([Guid]::NewGuid().ToString('N')) $reply $replyBody } 'cloud verifier rejects the wrong response nonce'
    Expect-Rejection { Assert-WorkAIResponse $mockServer 201 $nonce $reply $replyBody } 'cloud verifier rejects a changed HTTP status'
    Expect-Rejection { Assert-WorkAIResponse $clientPublic 200 $nonce $reply $replyBody } 'cloud verifier rejects a response signed by another server'
    Expect-Rejection { Assert-WorkAIResponse $mockServer 403 $nonce @{} $replyBody } 'cloud verifier rejects an unsigned proxy error'

    # Pure local installation fixtures; no Windows service or remote host is used.
    $script:FixtureListenerOpen = $false
    function Get-NetTCPConnection {
        [CmdletBinding()]param([string]$State)
        if ($script:FixtureListenerOpen) { [pscustomobject]@{LocalPort=18765} }
    }
    $reportedDiagnostic = Read-JsonFile (Join-Path $PSScriptRoot 'connection/installed-client-diagnostics-20261022.json')
    function New-PairingFixture([string]$Name,[switch]$FailReportWrite,[switch]$UseReportedKey) {
        $base = Join-Path $fixtureRoot $Name
        $installation = Join-Path $base 'original installation'
        $package = Join-Path $base 'package'
        foreach ($p in @($installation,$package,(Join-Path $installation 'public'),(Join-Path $installation 'jobs'))) { $null = [IO.Directory]::CreateDirectory($p) }
        $common = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'connection/Http.Common.ps1'))
        if ($FailReportWrite) { $common += "`nfunction Write-JsonFile { throw 'Injected fixture report-write failure.' }`n" }
        [IO.File]::WriteAllText((Join-Path $installation 'Http.Common.ps1'),$common,$script:Utf8)
        $oldPublic = $oldClient.ToXmlString($false)
        $oldPublicFile = $oldPublic
        if ($UseReportedKey) {
            $oldPublic = $reportedDiagnostic.public_key_xml
            $oldPublicFile = $oldPublic+"`n"
        }
        [IO.File]::WriteAllText((Join-Path $installation 'cloud-client-public.xml'),$oldPublicFile,$script:Utf8)
        Write-JsonFile (Join-Path $installation 'settings.json') @{schema='workai.comsol.http.settings.v1';configured=$true;base_url=$connection.base_url}
        $report = Read-JsonFile (Join-Path $PSScriptRoot 'connection/server-original.json')
        $report.cloud_client_public_key_sha256 = Get-BytesHash $script:Utf8.GetBytes($oldPublic)
        Write-JsonFile (Join-Path $installation 'public/http-connection-info.json') $report
        $spec = Read-JsonFile (Join-Path $PSScriptRoot 'pairing/pairing-manifest.json')
        $spec.previous_client_public_key_sha256 = $report.cloud_client_public_key_sha256
        $spec.http_common_sha256 = Get-FileHashText (Join-Path $installation 'Http.Common.ps1')
        Write-JsonFile (Join-Path $package 'pairing-manifest.json') $spec
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'pairing/cloud-client-public.xml') -Destination $package
        return [pscustomobject]@{Installation=$installation;Package=$package;OldClientHash=(Get-FileHashText (Join-Path $installation 'cloud-client-public.xml'));OldReportHash=(Get-FileHashText (Join-Path $installation 'public/http-connection-info.json'));SettingsHash=(Get-FileHashText (Join-Path $installation 'settings.json'))}
    }
    $fixture = New-PairingFixture 'successful migration'
    $null = Install-WorkAICloudPairing $fixture.Installation $fixture.Package
    $newHash = (Read-JsonFile (Join-Path $PSScriptRoot 'connection/client-identity.json')).public_key_sha256
    Check ((Get-FileHashText (Join-Path $fixture.Installation 'cloud-client-public.xml')) -ceq $newHash -and (Read-JsonFile (Join-Path $fixture.Installation 'public/http-connection-info.json')).cloud_client_public_key_sha256 -ceq $newHash) 'pairing fixture installs matching public key and report'
    Check ((Get-FileHashText (Join-Path $fixture.Installation 'settings.json')) -ceq $fixture.SettingsHash) 'pairing leaves original settings unchanged'
    $backups = @(Get-ChildItem -LiteralPath (Join-Path $fixture.Installation 'cloud-pairing-backups') -Directory)
    Check ($backups.Count -eq 1 -and (Get-FileHashText (Join-Path $backups[0].FullName 'cloud-client-public.xml')) -ceq $fixture.OldClientHash -and (Get-FileHashText (Join-Path $backups[0].FullName 'http-connection-info.json')) -ceq $fixture.OldReportHash) 'pairing preserves both original files in its backup'
    $after = Get-FileHashText (Join-Path $fixture.Installation 'public/http-connection-info.json')
    $null = Install-WorkAICloudPairing $fixture.Installation $fixture.Package
    Check ((Get-FileHashText (Join-Path $fixture.Installation 'public/http-connection-info.json')) -ceq $after -and @(Get-ChildItem -LiteralPath (Join-Path $fixture.Installation 'cloud-pairing-backups') -Directory).Count -eq 1) 'pairing is idempotent without changing identity or making another backup'

    $fixture = New-PairingFixture 'reported original key with trailing LF' -UseReportedKey
    $originalSpec = Read-JsonFile (Join-Path $PSScriptRoot 'pairing/pairing-manifest.json')
    Check ($fixture.OldClientHash -ceq $reportedDiagnostic.file_sha256 -and
        (Read-JsonFile (Join-Path $fixture.Package 'pairing-manifest.json')).previous_client_public_key_sha256 -ceq $originalSpec.previous_client_public_key_sha256 -and
        (Get-BytesHash ($script:Utf8.GetBytes($reportedDiagnostic.public_key_xml))) -ceq $originalSpec.previous_client_public_key_sha256) 'real reported LF fixture reproduces the exact raw hash and the unchanged original key pin'
    $null = Install-WorkAICloudPairing $fixture.Installation $fixture.Package
    Check ((Get-FileHashText (Join-Path $fixture.Installation 'cloud-client-public.xml')) -ceq $newHash -and
        (Read-JsonFile (Join-Path $fixture.Installation 'public/http-connection-info.json')).cloud_client_public_key_sha256 -ceq $newHash) 'real reported original key migrates successfully despite its trailing newline'
    $backups = @(Get-ChildItem -LiteralPath (Join-Path $fixture.Installation 'cloud-pairing-backups') -Directory)
    Check ($backups.Count -eq 1 -and
        (Get-FileHashText (Join-Path $backups[0].FullName 'cloud-client-public.xml')) -ceq $reportedDiagnostic.file_sha256 -and
        (Get-FileHashText (Join-Path $backups[0].FullName 'http-connection-info.json')) -ceq $fixture.OldReportHash -and
        (Get-FileHashText (Join-Path $fixture.Installation 'settings.json')) -ceq $fixture.SettingsHash) 'migration preserves the original newline bytes in backup and leaves settings unchanged'
    $after = Get-FileHashText (Join-Path $fixture.Installation 'public/http-connection-info.json')
    $newKeyPath = Join-Path $fixture.Installation 'cloud-client-public.xml'
    [IO.File]::AppendAllText($newKeyPath,"`n",$script:Utf8)
    $newFileHash = Get-FileHashText $newKeyPath
    $null = Install-WorkAICloudPairing $fixture.Installation $fixture.Package
    Check ((Get-FileHashText $newKeyPath) -ceq $newFileHash -and
        (Get-FileHashText (Join-Path $fixture.Installation 'public/http-connection-info.json')) -ceq $after -and
        @(Get-ChildItem -LiteralPath (Join-Path $fixture.Installation 'cloud-pairing-backups') -Directory).Count -eq 1) 'already paired key with trailing newline is a byte-preserving no-op'

    $fixture = New-PairingFixture 'BOM and whitespace on original key' -UseReportedKey
    [IO.File]::WriteAllText((Join-Path $fixture.Installation 'cloud-client-public.xml'),("`r`n "+$reportedDiagnostic.public_key_xml+" `r`n"),(New-Object Text.UTF8Encoding($true)))
    $encodedHash = Get-FileHashText (Join-Path $fixture.Installation 'cloud-client-public.xml')
    $null = Install-WorkAICloudPairing $fixture.Installation $fixture.Package
    $backups = @(Get-ChildItem -LiteralPath (Join-Path $fixture.Installation 'cloud-pairing-backups') -Directory)
    Check ((Get-FileHashText (Join-Path $fixture.Installation 'cloud-client-public.xml')) -ceq $newHash -and
        (Get-FileHashText (Join-Path $backups[0].FullName 'cloud-client-public.xml')) -ceq $encodedHash) 'boundary whitespace and a BOM preserve key identity and are backed up exactly'

    $fixture = New-PairingFixture 'different installed public key' -UseReportedKey
    $differentKeyPath = Join-Path $fixture.Installation 'cloud-client-public.xml'
    [IO.File]::WriteAllText($differentKeyPath,($mockServer.ToXmlString($false)+"`n"),$script:Utf8)
    $differentHash = Get-FileHashText $differentKeyPath
    Expect-Rejection { Install-WorkAICloudPairing $fixture.Installation $fixture.Package } 'a different RSA key with a trailing newline still fails the original pin'
    Check ((Get-FileHashText $differentKeyPath) -ceq $differentHash -and
        (Get-FileHashText (Join-Path $fixture.Installation 'public/http-connection-info.json')) -ceq $fixture.OldReportHash -and
        -not (Test-Path -LiteralPath (Join-Path $fixture.Installation 'cloud-pairing-backups'))) 'identity mismatch stops before any installation write or backup'

    $fixture = New-PairingFixture 'stale report with known key' -UseReportedKey
    $staleReportPath = Join-Path $fixture.Installation 'public/http-connection-info.json'
    $staleReport = Read-JsonFile $staleReportPath
    $staleReport.cloud_client_public_key_sha256 = $newHash
    Write-JsonFile $staleReportPath $staleReport
    $staleReportHash = Get-FileHashText $staleReportPath
    Expect-Rejection { Install-WorkAICloudPairing $fixture.Installation $fixture.Package } 'a report claiming another identity is still rejected'
    Check ((Get-FileHashText (Join-Path $fixture.Installation 'cloud-client-public.xml')) -ceq $fixture.OldClientHash -and
        (Get-FileHashText $staleReportPath) -ceq $staleReportHash -and
        -not (Test-Path -LiteralPath (Join-Path $fixture.Installation 'cloud-pairing-backups'))) 'stale report refusal preserves the original bytes'

    $fixture = New-PairingFixture 'changed candidate artifact' -UseReportedKey
    [IO.File]::AppendAllText((Join-Path $fixture.Package 'cloud-client-public.xml'),"`n",$script:Utf8)
    Expect-Rejection { Install-WorkAICloudPairing $fixture.Installation $fixture.Package } 'new public-key artifact still requires its exact original file checksum'
    Check ((Get-FileHashText (Join-Path $fixture.Installation 'cloud-client-public.xml')) -ceq $fixture.OldClientHash -and
        -not (Test-Path -LiteralPath (Join-Path $fixture.Installation 'cloud-pairing-backups'))) 'candidate checksum failure preserves the installation'

    $fixture = New-PairingFixture 'running listener' -UseReportedKey
    $script:FixtureListenerOpen = $true
    Expect-Rejection { Install-WorkAICloudPairing $fixture.Installation $fixture.Package } 'pairing refuses to change an active API listener'
    Check ((Get-FileHashText (Join-Path $fixture.Installation 'cloud-client-public.xml')) -ceq $fixture.OldClientHash) 'active-listener refusal preserves the installed key'
    $script:FixtureListenerOpen = $false
    $fixture = New-PairingFixture 'running job' -UseReportedKey
    $job = Join-Path $fixture.Installation 'jobs/comsol-00000000000000000000000000000001'
    $null = [IO.Directory]::CreateDirectory($job)
    Write-JsonFile (Join-Path $job 'status.json') @{phase='running'}
    Expect-Rejection { Install-WorkAICloudPairing $fixture.Installation $fixture.Package } 'pairing refuses to change an active or unresolved job'
    Check ((Get-FileHashText (Join-Path $fixture.Installation 'cloud-client-public.xml')) -ceq $fixture.OldClientHash) 'active-job refusal preserves the installed key'

    $fixture = New-PairingFixture 'write failure' -FailReportWrite -UseReportedKey
    Expect-Rejection { Install-WorkAICloudPairing $fixture.Installation $fixture.Package } 'pairing surfaces a simulated report-write failure'
    Check ((Get-FileHashText (Join-Path $fixture.Installation 'cloud-client-public.xml')) -ceq $fixture.OldClientHash -and (Get-FileHashText (Join-Path $fixture.Installation 'public/http-connection-info.json')) -ceq $fixture.OldReportHash) 'pairing rolls back both original files after a simulated write failure'
    Write-Output ('LOCAL CHECKS PASSED: '+$script:ChecksPassed)
    Write-Output 'NOT TESTED: Windows networking, Tailscale reachability, live authentication, license, or COMSOL execution.'
} finally {
    foreach ($rsa in @($client,$clientPublic,$mockServer,$oldClient)) { $rsa.Dispose() }
    Remove-Item -LiteralPath $fixtureRoot -Recurse -Force
}
