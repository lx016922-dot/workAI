$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'diagnostics/Export-PairingDiagnostics.ps1')
$tokens = $null; $errors = $null
$null = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'diagnostics/Export-PairingDiagnostics.ps1'),[ref]$tokens,[ref]$errors)
if ($errors.Count) { throw ($errors.Message -join ' | ') }
$fixture = Join-Path '/tmp' ('workai-diagnostics-check-'+[Guid]::NewGuid().ToString('N'))
$null = [IO.Directory]::CreateDirectory($fixture)
$null = [IO.Directory]::CreateDirectory((Join-Path $fixture 'public'))
$outputDirectory = Join-Path $fixture 'output'
$null = [IO.Directory]::CreateDirectory($outputDirectory)
$utf8 = New-Object Text.UTF8Encoding($false)
$utf8Bom = New-Object Text.UTF8Encoding($true)
$keyPath = Join-Path $fixture 'cloud-client-public.xml'
$publicKey = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'pairing/cloud-client-public.xml'))
$expected = Get-DiagnosticBytesHash $utf8.GetBytes($publicKey)
$passed = 0
function Assert-DiagnosticCheck([bool]$Condition,[string]$Label) {
    if (-not $Condition) { throw ('CHECK FAILED: '+$Label) }
    Write-Output ('PASS: '+$Label)
}
try {
    [IO.File]::WriteAllText($keyPath,$publicKey,$utf8Bom)
    $info = Get-PublicKeyDiagnostic $keyPath
    Assert-DiagnosticCheck ($info.encoding -ceq 'UTF-8 BOM' -and $info.file_sha256 -cne $expected -and $info.decoded_text_sha256 -ceq $expected -and $info.canonical_public_xml_sha256 -ceq $expected) 'BOM changes file hash while decoded and canonical public identity stay equal'
    $passed++
    [IO.File]::WriteAllText($keyPath,($publicKey+"`r`n"),$utf8)
    $info = Get-PublicKeyDiagnostic $keyPath
    Assert-DiagnosticCheck ($info.file_sha256 -cne $expected -and $info.trimmed_text_sha256 -ceq $expected -and $info.canonical_public_xml_sha256 -ceq $expected) 'trailing newline is diagnosed separately from RSA identity'
    $passed++
    [IO.File]::WriteAllText($keyPath,($publicKey.Replace('><',">`r`n<")),$utf8)
    $info = Get-PublicKeyDiagnostic $keyPath
    Assert-DiagnosticCheck ($info.decoded_text_sha256 -cne $expected -and $info.canonical_public_xml_sha256 -ceq $expected) 'formatted public XML retains the same normalized RSA identity'
    $passed++
    [IO.File]::WriteAllText($keyPath,$publicKey,[Text.Encoding]::Unicode)
    $info = Get-PublicKeyDiagnostic $keyPath
    Assert-DiagnosticCheck ($info.encoding -ceq 'UTF-16 LE BOM' -and $info.file_sha256 -cne $expected -and $info.decoded_text_sha256 -ceq $expected) 'UTF-16 encoding is recognized without changing expected fingerprints'
    $passed++
    $report = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'connection/server-original.json') | ConvertFrom-Json
    [IO.File]::WriteAllText($keyPath,$report.server_public_key_xml,$utf8)
    $info = Get-PublicKeyDiagnostic $keyPath
    Assert-DiagnosticCheck ($info.canonical_public_xml_sha256 -cne $expected) 'a genuinely different public key stays distinguishable'
    $passed++
    # A synthetic private-field marker is enough to prove rejection; no private
    # credential file is read by this test or by the diagnostic tool.
    [IO.File]::WriteAllText($keyPath,$publicKey.Replace('</RSAKeyValue>','<D>AQAB</D></RSAKeyValue>'),$utf8)
    $rejected = $false
    try { $null = Get-PublicKeyDiagnostic $keyPath } catch { $rejected = $true }
    Assert-DiagnosticCheck $rejected 'private-key fields are rejected before export'
    $passed++
    [IO.File]::WriteAllText($keyPath,('<!DOCTYPE RSAKeyValue [<!ENTITY x "external">]>'+$publicKey),$utf8)
    $rejected = $false
    try { $null = Get-PublicKeyDiagnostic $keyPath } catch { $rejected = $true }
    Assert-DiagnosticCheck $rejected 'XML document type declarations are rejected'
    $passed++

    [IO.File]::WriteAllText($keyPath,$publicKey,$utf8Bom)
    $report.cloud_client_public_key_sha256 = $expected
    $report | Add-Member -NotePropertyName unexpected_private_data -NotePropertyValue 'DIAGNOSTIC_TEST_SENTINEL_DO_NOT_EXPORT'
    [IO.File]::WriteAllText((Join-Path $fixture 'public/http-connection-info.json'),($report | ConvertTo-Json -Depth 10),$utf8)
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'connection/Http.Common.ps1') -Destination $fixture
    $before = @{}
    foreach ($name in @('cloud-client-public.xml','public/http-connection-info.json','Http.Common.ps1')) {
        $before[$name] = (Get-FileHash -LiteralPath (Join-Path $fixture $name) -Algorithm SHA256).Hash
    }
    $path = Export-WorkAIPairingDiagnostics $fixture $outputDirectory (Join-Path $PSScriptRoot 'diagnostics/pairing-manifest.json')
    $json = [IO.File]::ReadAllText($path)
    $result = $json | ConvertFrom-Json
    Assert-DiagnosticCheck ($result.matches_reported_client_any_representation -eq $true -and $result.matches_prepared_new_client_any_representation -eq $true -and $result.private_keys_read -eq $false -and -not $json.Contains('DIAGNOSTIC_TEST_SENTINEL_DO_NOT_EXPORT')) 'end-to-end export includes only allowlisted public diagnostics'
    $passed++
    foreach ($name in $before.Keys) {
        if ((Get-FileHash -LiteralPath (Join-Path $fixture $name) -Algorithm SHA256).Hash -cne $before[$name]) { throw ('Installation fixture changed: '+$name) }
    }
    Assert-DiagnosticCheck $true 'diagnostic export preserves every original installation file byte-for-byte'
    $passed++
    Write-Output ('DIAGNOSTIC CHECKS PASSED: '+$passed)
} finally { Remove-Item -LiteralPath $fixture -Recurse -Force }
