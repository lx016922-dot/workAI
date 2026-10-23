# Cloud-side helper. Reuses the uploaded server protocol without modifying it.
$ErrorActionPreference = 'Stop'
$protocolPath = Join-Path $PSScriptRoot 'connection/Http.Common.ps1'
if ((Get-FileHash -LiteralPath $protocolPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne '091cfcad0077b58903a4e9117fa85d8f56b1d3950350f0b974cfb0acddda9b41') {
    throw 'The recovered signing implementation changed; review it before use.'
}
. $protocolPath

function Read-WorkAIConnection([string]$Path) {
    $info = Read-JsonFile $Path
    if ($info.schema -cne 'workai.comsol.http.connection.v1') { throw 'Unknown connection schema.' }
    $expected = '760522526952265db4d29f69ed56dc513b59eaccd9c26336a1081831aa125889'
    if ($info.server_public_key_sha256 -cne $expected -or (Get-BytesHash $script:Utf8.GetBytes($info.server_public_key_xml)) -cne $expected) {
        throw 'Server identity differs from the public key supplied by the user.'
    }
    $uri = [Uri]$info.base_url
    if (-not $uri.IsAbsoluteUri -or $uri.Scheme -notin @('http','https') -or $uri.UserInfo -or $uri.Query -or $uri.Fragment -or $uri.AbsolutePath -ne '/') {
        throw 'Expected an HTTP(S) API origin without credentials, query, or path.'
    }
    return $info
}

function New-WorkAIRequestHeaders($Rsa,[string]$Method,[string]$Path,[byte[]]$Body) {
    if ($Method -cnotin @('GET','POST') -or $Path -cnotmatch '^/v1/[A-Za-z0-9_./-]+$' -or $Path.Contains('..')) { throw 'Invalid API method or path.' }
    if ($Body.Length -gt 65536 -or ($Method -ceq 'GET' -and $Body.Length)) { throw 'Invalid request body length.' }
    $timestamp = [string](Get-UnixTime)
    $nonce = [Guid]::NewGuid().ToString('N')
    $headers = @{'X-WorkAI-Date'=$timestamp; 'X-WorkAI-Nonce'=$nonce}
    if ($null -ne $Rsa) {
        $headers['X-WorkAI-Signature'] = Get-Signature $Rsa (Get-RequestText $Method $Path $timestamp $nonce (Get-BytesHash $Body))
    }
    return $headers
}

function Assert-WorkAIResponse($ServerRsa,[int]$Status,[string]$ExpectedNonce,$Headers,[byte[]]$Body) {
    foreach ($name in @('X-WorkAI-Nonce','X-WorkAI-Body-SHA256','X-WorkAI-Signature')) {
        if (-not $Headers.ContainsKey($name)) { throw ('Unverified HTTP '+$Status+' response: missing '+$name+'. It may be a proxy response.') }
    }
    if ($ExpectedNonce -cnotmatch '^[a-f0-9]{32}$' -or $Headers['X-WorkAI-Nonce'] -cne $ExpectedNonce) { throw 'Response nonce mismatch.' }
    $hash = Get-BytesHash $Body
    if ($Headers['X-WorkAI-Body-SHA256'] -cne $hash) { throw 'Response body checksum mismatch.' }
    $valid = $false
    try {
        $canonical = Get-ResponseText $Status $ExpectedNonce $hash
        $valid = $ServerRsa.VerifyData($script:Utf8.GetBytes($canonical),[Convert]::FromBase64String($Headers['X-WorkAI-Signature']),[Security.Cryptography.HashAlgorithmName]::SHA256,[Security.Cryptography.RSASignaturePadding]::Pkcs1)
    } catch { $valid = $false }
    if (-not $valid) { throw 'Server response signature is invalid.' }
}

function Invoke-WorkAIRequest($Connection,[string]$Method,[string]$Path,[byte[]]$Body,[string]$PrivateKeyPath,[switch]$Unsigned,[switch]$ReturnBytes) {
    $clientRsa = $null; $serverRsa = $null; $handler = $null; $http = $null; $message = $null; $response = $null
    try {
        if (-not $Unsigned) {
            $clientRsa = New-RsaFromXml ([IO.File]::ReadAllText($PrivateKeyPath))
            $identityPath = Join-Path $PSScriptRoot 'connection/client-identity.json'
            $identity = Read-JsonFile $identityPath
            if ((Get-BytesHash $script:Utf8.GetBytes($clientRsa.ToXmlString($false))) -cne $identity.public_key_sha256) { throw 'Local client identity mismatch; do not automatically rotate keys.' }
        }
        $serverRsa = New-RsaFromXml $Connection.server_public_key_xml
        $headers = New-WorkAIRequestHeaders $clientRsa $Method $Path $Body
        $handler = [Net.Http.HttpClientHandler]::new()
        $handler.AllowAutoRedirect = $false
        $handler.UseCookies = $false
        # Use the platform's proxy for the private Tailscale destination too.
        $uri = [Uri]($Connection.base_url.TrimEnd('/')+$Path)
        $proxyName = if ($uri.Scheme -eq 'https') {'https_proxy'} else {'http_proxy'}
        $proxyUrl = [Environment]::GetEnvironmentVariable($proxyName)
        if (-not $proxyUrl) { $proxyUrl = [Environment]::GetEnvironmentVariable($proxyName.ToUpperInvariant()) }
        if ($proxyUrl) { $handler.Proxy = [Net.WebProxy]::new($proxyUrl); $handler.UseProxy = $true }
        $http = [Net.Http.HttpClient]::new($handler)
        $http.Timeout = [TimeSpan]::FromSeconds(25)
        $http.MaxResponseContentBufferSize = 1048576
        $message = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::new($Method),$uri)
        $message.Content = [Net.Http.ByteArrayContent]::new($Body)
        $message.Content.Headers.ContentLength = $Body.Length
        if ($Body.Length) { $message.Content.Headers.ContentType = [Net.Http.Headers.MediaTypeHeaderValue]::new('application/json') }
        foreach ($name in $headers.Keys) { $null = $message.Headers.TryAddWithoutValidation($name,$headers[$name]) }
        $response = $http.SendAsync($message).GetAwaiter().GetResult()
        $bytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
        $replyHeaders = @{}
        foreach ($header in $response.Headers) {
            if ($header.Key -in @('X-WorkAI-Nonce','X-WorkAI-Body-SHA256','X-WorkAI-Signature')) {
                $values = @($header.Value)
                if ($values.Count -ne 1) { throw 'Repeated signature response header.' }
                $replyHeaders[$header.Key] = $values[0]
            }
        }
        $status = [int]$response.StatusCode
        Assert-WorkAIResponse $serverRsa $status $headers['X-WorkAI-Nonce'] $replyHeaders $bytes
        if ($ReturnBytes) {
            return [pscustomobject]@{http_status=$status;server_signature_verified=$true;path=$Path;body_bytes=$bytes}
        }
        return [pscustomobject]@{http_status=$status;server_signature_verified=$true;path=$Path;body=(ConvertFrom-WorkAIJson $script:Utf8.GetString($bytes))}
    } finally {
        foreach ($item in @($response,$message,$http,$handler,$clientRsa,$serverRsa)) { if ($null -ne $item) { $item.Dispose() } }
    }
}
