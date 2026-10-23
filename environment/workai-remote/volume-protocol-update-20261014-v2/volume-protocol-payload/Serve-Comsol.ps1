#Requires -Version 5.1
param([string]$Root=$PSScriptRoot)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Http.Common.ps1')
$Root = [IO.Path]::GetFullPath($Root)
$settingsPath = Join-Path $Root 'settings.json'
if (-not (Test-Path -LiteralPath $settingsPath -PathType Leaf)) {
    Write-Host 'HTTP SETUP REQUIRED: settings.json has not been created in this folder.' -ForegroundColor Yellow
    Write-Host ('Folder: '+$Root)
    Write-Host 'Right-click Setup-HTTP.cmd in this folder and select Run as administrator.'
    Write-Host 'Wait for HTTP SETUP COMPLETE, then run Start-HTTP.cmd again.'
    Write-Host 'If setup stops, send the Setup stopped error from the SETUP window.'
    exit 2
}
try {
    $settings = Read-JsonFile $settingsPath
    if ($settings.schema -ne 'workai.comsol.http.settings.v1') { throw 'Unknown settings format.' }
    if (-not $settings.PSObject.Properties['configured'] -or $settings.configured -isnot [bool] -or $settings.configured -ne $true) {
        throw 'Setup did not finish. Run Setup-HTTP.cmd as administrator and check its output.'
    }
    foreach ($required in @('cloud-client-public.xml','private/server-private.xml','public/http-connection-info.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root $required) -PathType Leaf)) {
            throw ('Setup output is missing: '+$required+'. Run Setup-HTTP.cmd and check its output.')
        }
    }
} catch {
    Write-Host ('HTTP STARTUP STOPPED: '+$_.Exception.Message) -ForegroundColor Yellow
    Write-Host ('Folder: '+$Root)
    Write-Host 'Copy the SETUP window error back in chat. No COMSOL task has been started.'
    exit 2
}
Assert-Template (Join-Path $Root 'template') (Read-JsonFile (Join-Path $Root 'bundle-manifest.json'))
$plateReady=Test-Path -LiteralPath (Join-Path $Root 'plate-manifest.json') -PathType Leaf
if($plateReady){. (Join-Path $Root 'Plate.Common.ps1');$plateManifest=Assert-PlateTemplate $Root}
$researchReady=Test-Path -LiteralPath (Join-Path $Root 'research-manifest.json') -PathType Leaf
if($researchReady){. (Join-Path $Root 'Research.Common.ps1');$researchManifest=Assert-ResearchTemplate $Root}
$clientRsa = New-RsaFromXml ([IO.File]::ReadAllText((Join-Path $Root 'cloud-client-public.xml')))
$serverRsa = New-RsaFromXml ([IO.File]::ReadAllText((Join-Path (Join-Path $Root 'private') 'server-private.xml')))
$seen = @{}
$listener = New-Object Net.HttpListener
$listener.Prefixes.Add($settings.base_url+'/')
try {
    $listener.TimeoutManager.EntityBody = [TimeSpan]::FromSeconds(15)
    $listener.TimeoutManager.IdleConnection = [TimeSpan]::FromSeconds(15)
} catch { } # Some .NET implementations do not expose these HTTP.sys settings.

function Send-Result($Context,[int]$Status,[string]$Nonce,$Value,[string]$File='') {
    $response = $Context.Response
    $stream = $null
    try {
        if ($File) {
            # Lock the exact file while hashing and streaming, avoiding a changed-file race.
            $stream = [IO.File]::Open($File,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
            if ($stream.Length -gt 536870912) { throw 'Artifact exceeds the 512 MiB transfer limit.' }
            $sha = [Security.Cryptography.SHA256]::Create()
            try { $hash = ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
            finally { $sha.Dispose() }
            $stream.Position = 0
            $length = $stream.Length
            $response.ContentType = 'application/octet-stream'
        } else {
            $bytes = $script:Utf8.GetBytes(($Value | ConvertTo-Json -Depth 20 -Compress))
            $hash = Get-BytesHash $bytes
            $length = $bytes.Length
            $response.ContentType = 'application/json; charset=utf-8'
        }
        $response.StatusCode = $Status
        $response.ContentLength64 = $length
        $response.Headers.Add('Cache-Control','no-store')
        $response.Headers.Add('X-WorkAI-Nonce',$Nonce)
        $response.Headers.Add('X-WorkAI-Body-SHA256',$hash)
        $response.Headers.Add('X-WorkAI-Signature',(Get-Signature $serverRsa (Get-ResponseText $Status $Nonce $hash)))
        if ($stream) { $stream.CopyTo($response.OutputStream) }
        else { $response.OutputStream.Write($bytes,0,$bytes.Length) }
    } finally { if ($stream) { $stream.Dispose() }; $response.Close() }
}

try {
    $listener.Start()
    $public = Read-JsonFile (Join-Path (Join-Path $Root 'public') 'http-connection-info.json')
    $public.listener_started_at_utc = [DateTime]::UtcNow.ToString('o')
    Write-JsonFile (Join-Path (Join-Path $Root 'public') 'http-connection-info.json') $public
    Write-Host 'WORKAI COMSOL HTTP API LISTENING' -ForegroundColor Green
    Write-Host $settings.base_url
    Write-Host ('Send only public\http-connection-info.json back in chat.')
    Write-Host 'Leave this window open. Ctrl+C stops the API; a running solver continues.'
    Write-Host 'Cloud connectivity, COMSOL license and model execution have not been tested yet.'
    while ($listener.IsListening) {
        # Polling lets Ctrl+C interrupt PowerShell without blocking forever in GetContext().
        $pending = $listener.GetContextAsync()
        while (-not $pending.IsCompleted) { Start-Sleep -Milliseconds 100 }
        $context = $pending.GetAwaiter().GetResult()
        $request = $context.Request
        $nonce = [string]$request.Headers['X-WorkAI-Nonce']
        if ($nonce -notmatch '^[a-f0-9]{32}$') { $nonce = '' }
        try {
            $path = $request.RawUrl
            if ($path -notmatch '^/v1/[A-Za-z0-9_./-]+$' -or $path.Contains('..') -or $request.HttpMethod -notin @('GET','POST')) {
                Send-Result $context 404 $nonce @{error='Unknown endpoint.'}; continue
            }
            if ($request.ContentLength64 -lt 0 -or $request.ContentLength64 -gt 65536) {
                Send-Result $context 413 $nonce @{error='Use a Content-Length of at most 65536 bytes.'}; continue
            }
            if (-not $request.Headers['X-WorkAI-Signature']) {
                Send-Result $context 401 $nonce @{error='Signed requests required.'}; continue
            }
            $body = New-Object byte[] ([int]$request.ContentLength64)
            $offset = 0
            while ($offset -lt $body.Length) {
                $read = $request.InputStream.Read($body,$offset,$body.Length-$offset)
                if ($read -le 0) { throw 'Incomplete request body.' }
                $offset += $read
            }
            if (-not (Test-RequestSignature $clientRsa $request.HttpMethod $path $request.Headers['X-WorkAI-Date'] $nonce $request.Headers['X-WorkAI-Signature'] $body $seen)) {
                Send-Result $context 401 $nonce @{error='Invalid signature, replayed nonce, or clock differs by over 5 minutes.'}; continue
            }
            if ($request.HttpMethod -eq 'GET' -and $body.Length -ne 0) { throw 'GET body is not supported.' }
            if ($request.HttpMethod -eq 'GET' -and $path -ceq '/v1/research/health') {
                if(-not $researchReady){Send-Result $context 409 $nonce @{error='Research update not installed.'};continue}
                $researchManifest=Assert-ResearchTemplate $Root
                Send-Result $context 200 $nonce @{schema='workai.comsol.research.health.v1';status='ready_for_job';source_sha256=$researchManifest.files.'research-template/UltraResearch.java';live_logs=$true;model_run_proven_by_health=$false};continue
            }
            if ($researchReady -and $request.HttpMethod -eq 'GET' -and $path -cmatch '^/v1/jobs/(comsol-[a-f0-9]{32})/logs/(worker.log|solver.log|compile.log)$') {
                Send-Result $context 200 $nonce (Get-WorkAILiveLog $Root $Matches[1] $Matches[2]);continue
            }
            if ($researchReady -and $request.HttpMethod -eq 'POST' -and $path -ceq '/v1/research/revisions') {
                Send-Result $context 200 $nonce (Update-ResearchModel $Root (ConvertFrom-WorkAIJson ($script:Utf8.GetString($body))));continue
            }
            if ($request.HttpMethod -eq 'POST' -and $path -ceq '/v1/research/jobs') {
                if(-not $researchReady){Send-Result $context 409 $nonce @{error='Research update not installed.'};continue}
                $submission=ConvertFrom-WorkAIJson ($script:Utf8.GetString($body));$new=New-ResearchJobFiles $Root $submission
                if($new.Created){try{
                    $powershell=Join-Path $PSHOME 'powershell.exe';if(-not(Test-Path $powershell)){$powershell=Join-Path $PSHOME 'pwsh'}
                    if($Root.Contains('"')){throw 'Unsupported quote in installation path.'}
                    $arguments='-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "'+(Join-Path $new.Directory 'Run-ResearchJob.ps1')+'" -Root "'+$Root+'" -JobId '+$submission.job_id
                    $process=Start-Process -FilePath $powershell -ArgumentList $arguments -PassThru
                    Write-JsonFile (Join-Path $new.Directory 'launch.json') @{pid=$process.Id;started_at_utc=[DateTime]::UtcNow.ToString('o')}
                }catch{Write-JsonFile (Join-Path $new.Directory 'status.json') @{job_id=$submission.job_id;kind=$submission.kind;phase='failed';error=$_.Exception.Message;updated_at_utc=[DateTime]::UtcNow.ToString('o')};throw}}
                Send-Result $context 202 $nonce (Get-JobStatus $Root $submission.job_id);continue
            }
            if ($request.HttpMethod -eq 'GET' -and $path -ceq '/v1/volume/health') {
                if(-not $plateReady){Send-Result $context 409 $nonce @{error='Plate interface missing.'};continue}
                $volumeManifest=Assert-PlateTemplate $Root
                $modelReady=([IO.File]::ReadAllText((Join-Path $Root 'plate-template/UltraPlate.java'))).Contains('WORKAI_VOLUME_DISPLACEMENT_V1')
                Send-Result $context 200 $nonce @{schema='workai.comsol.volume.health.v1';protocol_ready=$true;model_ready=$modelReady;source_sha256=$volumeManifest.files.'UltraPlate.java';model_run_proven_by_health=$false};continue
            }
            if ($request.HttpMethod -eq 'GET' -and $path -ceq '/v1/plate/health') {
                if(-not $plateReady){Send-Result $context 409 $nonce @{error='Reviewed 3-D plate update is not installed.'};continue}
                $plateManifest=Assert-PlateTemplate $Root
                Send-Result $context 200 $nonce @{schema='workai.comsol.plate.health.v1';status='ready_for_job';source_sha256=$plateManifest.files.'UltraPlate.java';scope='finite_3d_isotropic_plate_open_surface_notch';kinds=@('plate_smoke','plate_reference');signed_model_updates=$true;model_run_proven_by_health=$false};continue
            }
            if ($request.HttpMethod -eq 'POST' -and $path -ceq '/v1/plate/revisions') {
                if(-not $plateReady){Send-Result $context 409 $nonce @{error='Plate model publication is not installed.'};continue}
                $submission=ConvertFrom-WorkAIJson ($script:Utf8.GetString($body))
                Send-Result $context 200 $nonce (Update-PlateModel $Root $submission);continue
            }
            if ($request.HttpMethod -eq 'GET' -and $path -cmatch '^/v1/plate/revisions/(plate-[a-f0-9]{32})$') {
                $revision=Join-Path (Join-Path $Root 'model-revisions') $Matches[1]
                if(-not(Test-Path -LiteralPath (Join-Path $revision 'status.json'))){Send-Result $context 404 $nonce @{error='Model revision not found.'};continue}
                Send-Result $context 200 $nonce (Read-JsonFile (Join-Path $revision 'status.json'));continue
            }
            if ($request.HttpMethod -eq 'POST' -and $path -ceq '/v1/plate/jobs') {
                if(-not $plateReady){Send-Result $context 409 $nonce @{error='Reviewed 3-D plate update is not installed.'};continue}
                $submission=ConvertFrom-WorkAIJson ($script:Utf8.GetString($body));$new=New-PlateJobFiles $Root $submission
                if($new.Created){try{
                    $powershell=Join-Path $PSHOME 'powershell.exe';if(-not(Test-Path $powershell)){$powershell=Join-Path $PSHOME 'pwsh'}
                    if($Root.Contains('"')){throw 'Unsupported quote in installation path.'}
                    $arguments='-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "'+(Join-Path $Root 'Run-PlateJob.ps1')+'" -Root "'+$Root+'" -JobId '+$submission.job_id
                    $process=Start-Process -FilePath $powershell -ArgumentList $arguments -PassThru
                    Write-JsonFile (Join-Path $new.Directory 'launch.json') @{pid=$process.Id;started_at_utc=[DateTime]::UtcNow.ToString('o')}
                }catch{Write-JsonFile (Join-Path $new.Directory 'status.json') @{job_id=$submission.job_id;kind=$submission.kind;phase='failed';error=$_.Exception.Message;updated_at_utc=[DateTime]::UtcNow.ToString('o')};throw}}
                Send-Result $context 202 $nonce (Get-JobStatus $Root $submission.job_id);continue
            }
            if ($request.HttpMethod -eq 'GET' -and $path -ceq '/v1/health') {
                $template = Read-JsonFile (Join-Path (Join-Path $Root 'template') 'request.json')
                $binary = Get-Item -LiteralPath (Join-Path $settings.comsol_bin 'comsolbatch.exe')
                Send-Result $context 200 $nonce ([ordered]@{
                    schema='workai.comsol.http.health.v1'; status='ready_for_job'; server_time_utc=[DateTime]::UtcNow.ToString('o');
                    comsol_file_version=$binary.VersionInfo.FileVersion; case_sha256=$template.case_sha256; source_sha256=$template.source_sha256;
                    scope='fixed_reviewed_1d_case'; license_tested=$false; model_run_proven_by_health=$false
                }); continue
            }
            if ($request.HttpMethod -eq 'POST' -and $path -ceq '/v1/jobs') {
                $submission = ConvertFrom-WorkAIJson ($script:Utf8.GetString($body))
                $new = New-JobFiles $Root $submission
                if ($new.Created) {
                    try {
                        $powershell = Join-Path $PSHOME 'powershell.exe'
                        if (-not (Test-Path $powershell)) { $powershell = Join-Path $PSHOME 'pwsh' }
                        if ($Root.Contains('"')) { throw 'Unsupported quote in local installation path.' }
                        $arguments = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "'+(Join-Path $Root 'Run-Job.ps1')+'" -Root "'+$Root+'" -JobId '+$submission.job_id
                        $process = Start-Process -FilePath $powershell -ArgumentList $arguments -PassThru
                        Write-JsonFile (Join-Path $new.Directory 'launch.json') @{pid=$process.Id; started_at_utc=[DateTime]::UtcNow.ToString('o')}
                    } catch {
                        Write-JsonFile (Join-Path $new.Directory 'status.json') @{job_id=$submission.job_id; kind=$submission.kind; phase='failed'; error=$_.Exception.Message; updated_at_utc=[DateTime]::UtcNow.ToString('o')}
                        throw
                    }
                }
                Send-Result $context 202 $nonce (Get-JobStatus $Root $submission.job_id); continue
            }
            if ($request.HttpMethod -eq 'GET' -and $path -cmatch '^/v1/jobs/(comsol-[a-f0-9]{32})$') {
                $id = $Matches[1]
                if (-not (Test-Path (Join-Path (Join-Path $Root 'jobs') $id))) { Send-Result $context 404 $nonce @{error='Job not found.'}; continue }
                Send-Result $context 200 $nonce (Get-JobStatus $Root $id); continue
            }
            if ($request.HttpMethod -eq 'GET' -and $path -cmatch '^/v1/jobs/(comsol-[a-f0-9]{32})/files/([a-z.]+)$') {
                $id = $Matches[1]; $name = $Matches[2]
                if ($name -cnotin $script:AllowedArtifacts) { Send-Result $context 404 $nonce @{error='Artifact is not allowlisted.'}; continue }
                $status = Get-JobStatus $Root $id
                if ($status.phase -notin @('completed','failed')) { Send-Result $context 409 $nonce @{error='Artifacts are available after the job finishes.'}; continue }
                $file = Join-Path (Join-Path (Join-Path $Root 'jobs') $id) $name
                if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { Send-Result $context 404 $nonce @{error='Artifact not found.'}; continue }
                Send-Result $context 200 $nonce $null $file; continue
            }
            Send-Result $context 404 $nonce @{error='Unknown endpoint.'}
        } catch {
            try { Send-Result $context 400 $nonce @{error=$_.Exception.Message} }
            catch { Write-Host 'HTTP request ended before a signed response could be delivered.' }
        }
    }
} finally {
    $listener.Close(); $clientRsa.Dispose(); $serverRsa.Dispose()
}
