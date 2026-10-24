[CmdletBinding()]
param(
    [ValidateSet('health','submit','status','log','download')][string]$Action='health',
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$SourceFile='', [string]$SceneFile='', [string]$ParametersFile='',
    [string]$JobId='', [string]$FileName='solver.log', [string]$OutputFile=''
)
$ErrorActionPreference='Stop'
. (Join-Path $Root 'WorkAI.Client.Common.ps1')
$validator=Join-Path $Root 'recovered-research-20261022/Research.Common.ps1'
if ((Get-FileHashText $validator) -cne '4a20b13f3410cfc37935860a653deca6d4c15cf23fdad109123dc400f7fa8748') { throw 'Research validator changed.' }
. $validator
$connection=Read-WorkAIConnection (Join-Path $Root 'connection/server-paired-20261022.json')
$key=Join-Path $Root 'private/client-private.xml'

# ResponseHeadersRead permits the server's existing, signed 512 MiB artifact
# protocol without changing the original helper's 1 MiB JSON limit. Verify the
# signature before downloading, then the exact byte count and SHA-256 before
# making an artifact available. Never follow redirects or disable proxy/trust.
function Save-VerifiedArtifact([string]$Path,[string]$Destination,$Expected) {
    $client=$null;$server=$null;$handler=$null;$http=$null;$message=$null;$response=$null;$input=$null;$output=$null;$hasher=$null;$cts=$null
    $temporary=$Destination+'.partial-'+[Guid]::NewGuid().ToString('N')
    try {
        if (Test-Path -LiteralPath $Destination) { throw 'Destination already exists; verify or choose a new path.' }
        $client=New-RsaFromXml ([IO.File]::ReadAllText($key))
        $identity=Read-JsonFile (Join-Path $Root 'connection/client-identity.json')
        if ((Get-BytesHash $script:Utf8.GetBytes($client.ToXmlString($false))) -cne $identity.public_key_sha256) { throw 'Client identity changed.' }
        $server=New-RsaFromXml $connection.server_public_key_xml
        $headers=New-WorkAIRequestHeaders $client 'GET' $Path ([byte[]]@())
        $uri=[Uri]($connection.base_url.TrimEnd('/')+$Path)
        $handler=[Net.Http.HttpClientHandler]::new();$handler.AllowAutoRedirect=$false;$handler.UseCookies=$false
        $proxyName=if ($uri.Scheme -eq 'https') {'https_proxy'} else {'http_proxy'}
        $proxy=[Environment]::GetEnvironmentVariable($proxyName)
        if (-not $proxy) { $proxy=[Environment]::GetEnvironmentVariable($proxyName.ToUpperInvariant()) }
        if ($proxy) { $handler.Proxy=[Net.WebProxy]::new($proxy);$handler.UseProxy=$true }
        $http=[Net.Http.HttpClient]::new($handler);$http.Timeout=[TimeSpan]::FromSeconds(1200)
        $cts=[Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds(1200))
        $message=[Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get,$uri)
        $message.Content=[Net.Http.ByteArrayContent]::new([byte[]]@());$message.Content.Headers.ContentLength=0
        foreach ($name in $headers.Keys) { $null=$message.Headers.TryAddWithoutValidation($name,$headers[$name]) }
        $response=$http.SendAsync($message,[Net.Http.HttpCompletionOption]::ResponseHeadersRead,$cts.Token).GetAwaiter().GetResult()
        $reply=@{}
        foreach ($header in $response.Headers) {
            if ($header.Key -in @('X-WorkAI-Nonce','X-WorkAI-Body-SHA256','X-WorkAI-Signature')) {
                $values=@($header.Value);if ($values.Count -ne 1) { throw 'Repeated signed header.' };$reply[$header.Key]=$values[0]
            }
        }
        if ($reply['X-WorkAI-Nonce'] -cne $headers['X-WorkAI-Nonce'] -or $reply['X-WorkAI-Body-SHA256'] -cnotmatch '^[a-f0-9]{64}$') { throw 'Missing signature or response nonce mismatch.' }
        $status=[int]$response.StatusCode
        $canonical=Get-ResponseText $status $headers['X-WorkAI-Nonce'] $reply['X-WorkAI-Body-SHA256']
        if (-not $server.VerifyData($script:Utf8.GetBytes($canonical),[Convert]::FromBase64String($reply['X-WorkAI-Signature']),[Security.Cryptography.HashAlgorithmName]::SHA256,[Security.Cryptography.RSASignaturePadding]::Pkcs1)) { throw 'Invalid server signature.' }
        $length=$response.Content.Headers.ContentLength
        if ($status -ne 200 -or $null -eq $length -or $length -lt 0 -or $length -gt 536870912) { throw ('Artifact status/size rejected: HTTP '+$status) }
        if ($length -ne $Expected.bytes -or $reply['X-WorkAI-Body-SHA256'] -cne $Expected.sha256) { throw 'Artifact differs from signed job manifest.' }
        $input=$response.Content.ReadAsStreamAsync($cts.Token).GetAwaiter().GetResult()
        $output=[IO.File]::Open($temporary,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        $hasher=[Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
        $buffer=[byte[]]::new(1048576);$count=[long]0
        while (($read=$input.ReadAsync($buffer,0,$buffer.Length,$cts.Token).GetAwaiter().GetResult()) -gt 0) {
            $count+=$read;if ($count -gt $length) { throw 'Oversized response.' }
            $hasher.AppendData($buffer,0,$read);$output.Write($buffer,0,$read)
        }
        $hash=([BitConverter]::ToString($hasher.GetHashAndReset())).Replace('-','').ToLowerInvariant()
        if ($count -ne $length -or $hash -cne $Expected.sha256) { throw 'Truncated or corrupted artifact.' }
        $output.Flush($true);$output.Dispose();$output=$null
        [IO.File]::Move($temporary,$Destination)
        return [ordered]@{http_status=$status;server_signature_verified=$true;manifest_verified=$true;path=$Path;bytes=$count;sha256=$hash;file=$Destination}
    } finally {
        foreach ($item in @($output,$input,$response,$message,$http,$handler,$client,$server,$hasher,$cts)) { if ($null -ne $item) { $item.Dispose() } }
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    }
}

if ($Action -eq 'health') {
    $result=Invoke-WorkAIRequest $connection 'GET' '/v1/research/health' ([byte[]]@()) $key
} else {
    Assert-JobId $JobId
    $jobPath='/v1/jobs/'+$JobId
    if ($Action -eq 'submit') {
        $sourceHash=Get-FileHashText $SourceFile
        $sceneText=[IO.File]::ReadAllText($SceneFile);$parametersText=[IO.File]::ReadAllText($ParametersFile)
        Assert-ResearchParameters (ConvertFrom-WorkAIJson $parametersText)
        $scene=ConvertFrom-WorkAIJson $sceneText
        if ($scene.schema -cne 'workai.comsol.graphwave3d.scene.v1') { throw 'Unexpected 3D scene schema.' }
        $health=Invoke-WorkAIRequest $connection 'GET' '/v1/research/health' ([byte[]]@()) $key
        if ($health.http_status -ne 200 -or $health.body.status -cne 'ready_for_job' -or $health.body.source_sha256 -cne $sourceHash) { throw 'Installed COMSOL reference source differs.' }
        $directory=Join-Path (Join-Path $Root 'requests') $JobId
        if (Test-Path -LiteralPath $directory) { throw 'Request ID already used. Query status; do not repeat POST.' }
        $submission=[ordered]@{job_id=$JobId;kind='research_reference';prepared_at_utc=[DateTime]::UtcNow.ToString('o');scene_sha256=(Get-BytesHash $script:Utf8.GetBytes($sceneText));source_sha256=$sourceHash;parameters_sha256=(Get-BytesHash $script:Utf8.GetBytes($parametersText));scene_json=$sceneText;parameters_json=$parametersText}
        $body=$script:Utf8.GetBytes(($submission | ConvertTo-Json -Depth 30 -Compress))
        if ($body.Length -gt 65536) { throw 'Request budget exceeded.' }
        $null=[IO.Directory]::CreateDirectory($directory);Write-JsonFile (Join-Path $directory 'submission.json') $submission
        try { $result=Invoke-WorkAIRequest $connection 'POST' '/v1/research/jobs' $body $key }
        catch { Write-JsonFile (Join-Path $directory 'uncertain.json') @{error=$_.Exception.Message;next_action='Query the same job ID; never automatically repeat POST.'};throw }
        Write-JsonFile (Join-Path $directory 'response.json') $result
        if ($result.http_status -ne 202) { $result | ConvertTo-Json -Depth 30;exit 2 }
    } elseif ($Action -eq 'log') {
        if ($FileName -cnotin @('solver.log','worker.log','compile.log')) { throw 'Unsupported live log.' }
        $result=Invoke-WorkAIRequest $connection 'GET' ($jobPath+'/logs/'+$FileName) ([byte[]]@()) $key
        if ($result.http_status -eq 200 -and $result.body.PSObject.Properties['tail_base64'] -and $result.body.tail_base64) {
            $result.body | Add-Member -NotePropertyName tail_utf8 -NotePropertyValue ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($result.body.tail_base64)))
            $result.body.PSObject.Properties.Remove('tail_base64')
        }
    } else {
        $result=Invoke-WorkAIRequest $connection 'GET' $jobPath ([byte[]]@()) $key
        if ($result.http_status -ne 200 -or $result.body.job_id -cne $JobId) { throw 'Job status mismatch.' }
        if ($Action -eq 'download') {
            if ($FileName -cnotmatch '^[a-z.]+$' -or $FileName.Contains('..') -or -not $OutputFile) { throw 'Invalid artifact destination.' }
            $matches=@($result.body.artifacts | Where-Object {$_.name -ceq $FileName})
            if ($matches.Count -ne 1) { throw 'Artifact missing from signed status manifest.' }
            $result=Save-VerifiedArtifact ($jobPath+'/files/'+$FileName) $OutputFile $matches[0]
        }
    }
}
if ($OutputFile -and $Action -ne 'download') { Write-JsonFile $OutputFile $result }
if ($Action -eq 'log' -and $result.body.PSObject.Properties['tail_utf8'] -and $result.body.tail_utf8.Length -gt 3200) {
    $result.body.tail_utf8=$result.body.tail_utf8.Substring($result.body.tail_utf8.Length-3200)
}
$result | ConvertTo-Json -Depth 30
