$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'WorkAI.Client.Common.ps1')
$privatePath = Join-Path $PSScriptRoot 'private/client-private.xml'
$publicPath = Join-Path $PSScriptRoot 'pairing/cloud-client-public.xml'
$identityPath = Join-Path $PSScriptRoot 'connection/client-identity.json'
$rsa = $null
try {
    if (Test-Path -LiteralPath $privatePath) {
        $rsa = New-RsaFromXml ([IO.File]::ReadAllText($privatePath))
    } else {
        if ((Test-Path -LiteralPath $publicPath) -or (Test-Path -LiteralPath $identityPath)) { throw 'An identity already exists but its private key is missing; do not silently generate a replacement.' }
        $null = [IO.Directory]::CreateDirectory((Split-Path $privatePath -Parent))
        $rsa = [Security.Cryptography.RSA]::Create(3072)
        # Run this helper with umask 077. The private value is never printed.
        [IO.File]::WriteAllText($privatePath,$rsa.ToXmlString($true),$script:Utf8)
    }
    $public = $rsa.ToXmlString($false)
    $sha = Get-BytesHash $script:Utf8.GetBytes($public)
    $null = [IO.Directory]::CreateDirectory((Split-Path $publicPath -Parent))
    if (Test-Path -LiteralPath $publicPath) {
        if ((Get-FileHashText $publicPath) -cne $sha) { throw 'Existing public key differs from the local private key.' }
    } else { [IO.File]::WriteAllText($publicPath,$public,$script:Utf8) }
    if (Test-Path -LiteralPath $identityPath) {
        if ((Read-JsonFile $identityPath).public_key_sha256 -cne $sha) { throw 'Existing identity fingerprint differs.' }
    } else {
        Write-JsonFile $identityPath ([ordered]@{schema='workai.cloud.client.identity.v1';public_key_sha256=$sha;created_at_utc=[DateTime]::UtcNow.ToString('o');private_key_file='private/client-private.xml';public_key_file='pairing/cloud-client-public.xml'})
    }
    Write-Output ('Client public key SHA-256: '+$sha)
} finally { if ($null -ne $rsa) { $rsa.Dispose() } }
