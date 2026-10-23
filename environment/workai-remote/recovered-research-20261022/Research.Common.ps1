#Requires -Version 5.1
# Additive research protocol. Does not change the running 1-D/3-D worker files.
function Assert-ResearchTemplate($Root) {
    $manifest=Read-JsonFile (Join-Path $Root 'research-manifest.json')
    $names=@('research-template/UltraResearch.java','research-template/run-research.ps1','Research.Common.ps1','Run-ResearchJob.ps1')
    if($manifest.schema -cne 'workai.comsol.research.bundle.v1' -or @($manifest.files.PSObject.Properties).Count -ne $names.Count){throw 'Invalid research manifest.'}
    foreach($name in $names){if(-not $manifest.files.PSObject.Properties[$name] -or (Get-FileHashText (Join-Path $Root $name)) -cne $manifest.files.PSObject.Properties[$name].Value){throw ('Research template changed: '+$name)}}
    return $manifest
}
function Assert-ResearchParameters($Parameters) {
    $fixed=@('schema','scenario','boundary','length_mm','height_mm','axisymmetric','outer_radius_mm','rho','cl','ct','conductivity','heat_capacity','expansion','thermal','duration_us','sample_step_us','relative_tolerance','element_order','element_budget','field_nx','field_ny','field_frames','notch_enabled','notch_x_mm','notch_width_mm','notch_depth_mm','notch_bottom','shot_count','receiver_count','coarse_h_mm','fine_h_mm','max_step_us','bandwidth_estimate_mhz')
    foreach($name in $fixed){if(-not $Parameters.PSObject.Properties[$name]){throw ('Missing research parameter: '+$name)}}
    if($Parameters.schema -cne 'workai.comsol.research.case.v1' -or $Parameters.scenario -cnotin @('guided_plate','guided_pipe','weld_array','laser_surface') -or $Parameters.boundary -cnotin @('free','side_impedance')){throw 'Research type not supported.'}
    if(@($Parameters.PSObject.Properties).Count -gt 1800){throw 'Too many parameters.'}
    foreach($item in $Parameters.PSObject.Properties){
        if($item.Name -in @('schema','scenario','boundary')){continue}
        if($item.Name -notin $fixed -and $item.Name -cnotmatch '^(tx([0-9]|[1-5][0-9]|6[0-3])_(kind|x_mm|radius_mm|delay_us|phase_deg|cycles|amplitude_pa|direction|frequency_count|f[0-7]|elements|pitch_mm|angle_deg|wedge_incidence_deg|wedge_cl|pulse_us|absorbed_j_per_m|gaussian_integral_m)|rx([0-9]|[12][0-9]|3[01])_x_mm)$'){throw ('Unknown numeric parameter: '+$item.Name)}
        if($item.Value -is [bool] -or $item.Value -is [string] -or $null -eq $item.Value){throw 'Numeric parameters cannot contain expressions.'}
        $value=[double]$item.Value
        if([double]::IsNaN($value) -or [double]::IsInfinity($value) -or [Math]::Abs($value) -gt 1e12){throw 'Invalid finite numeric parameter.'}
    }
    if($Parameters.shot_count -lt 1 -or $Parameters.shot_count -gt 64 -or $Parameters.receiver_count -lt 1 -or $Parameters.receiver_count -gt 32 -or $Parameters.element_budget -gt 40000 -or $Parameters.element_order -ne 4){throw 'Research job exceeds resource limits.'}
    if($Parameters.max_step_us -le 0 -or $Parameters.sample_step_us -le 0 -or $Parameters.fine_h_mm -le 0 -or $Parameters.duration_us -le 0){throw 'Nonpositive discretization.'}
    if($Parameters.duration_us/($Parameters.max_step_us*.5) -gt 100000.1 -or $Parameters.shot_count*$Parameters.receiver_count*($Parameters.duration_us/$Parameters.sample_step_us+1) -gt 2000000){throw 'Research time/export budget exceeded.'}
    if(2*$Parameters.length_mm*$Parameters.height_mm/($Parameters.fine_h_mm*$Parameters.fine_h_mm) -gt 40000){throw 'Estimated research mesh exceeds budget.'}
}
function New-ResearchJobFiles($Root,$Submission) {
    $expected=@('job_id','kind','prepared_at_utc','scene_sha256','source_sha256','parameters_sha256','scene_json','parameters_json')
    $actual=@($Submission.PSObject.Properties.Name)
    if($actual.Count -ne $expected.Count -or @($actual | Where-Object {$_ -notin $expected}).Count){throw 'Unexpected research submission fields.'}
    Assert-JobId $Submission.job_id
    if($Submission.kind -cne 'research_reference'){throw 'Only research_reference is supported.'}
    foreach($k in @('scene_sha256','source_sha256','parameters_sha256')){if($Submission.$k -cnotmatch '^[a-f0-9]{64}$'){throw 'Invalid identity.'}}
    # PS 5.1 keeps JSON dates as strings; PS 7.4 converts them to DateTime.
    $preparedText=$Submission.prepared_at_utc
    if($preparedText -is [DateTime]){$preparedText=$preparedText.ToUniversalTime().ToString('o')}
    if($preparedText -notmatch '(Z|\+00:00)$'){throw 'UTC timestamp required.'}
    $prepared=[DateTimeOffset]::Parse($preparedText,[Globalization.CultureInfo]::InvariantCulture)
    if($prepared -gt [DateTimeOffset]::UtcNow.AddMinutes(5) -or $prepared -lt [DateTimeOffset]::UtcNow.AddDays(-1)){throw 'Prepare a fresh job.'}
    if((Get-BytesHash $script:Utf8.GetBytes($Submission.scene_json)) -cne $Submission.scene_sha256 -or (Get-BytesHash $script:Utf8.GetBytes($Submission.parameters_json)) -cne $Submission.parameters_sha256){throw 'Research payload digest differs.'}
    $parameters=ConvertFrom-WorkAIJson $Submission.parameters_json;Assert-ResearchParameters $parameters
    $manifest=Assert-ResearchTemplate $Root
    if($Submission.source_sha256 -cne $manifest.files.'research-template/UltraResearch.java'){throw 'Research source differs from installed version.'}
    $jobs=Join-Path $Root 'jobs';$job=Join-Path $jobs $Submission.job_id
    if(Test-Path -LiteralPath $job){
        $old=Read-JsonFile (Join-Path $job 'submission.json');foreach($k in $expected){if($old.$k -cne $Submission.$k){throw 'Job ID belongs to a different request.'}}
        return [pscustomobject]@{Directory=$job;Created=$false}
    }
    foreach($dir in @(Get-ChildItem -LiteralPath $jobs -Directory)){
        $state=Read-JsonFile (Join-Path $dir.FullName 'status.json');if($state.phase -notin @('completed','failed')){throw 'Another COMSOL job is active or unresolved; it was preserved.'}
    }
    $null=[IO.Directory]::CreateDirectory($job)
    Write-JsonFile (Join-Path $job 'submission.json') $Submission
    Write-JsonFile (Join-Path $job 'request.json') $Submission
    $lines=New-Object 'System.Collections.Generic.List[string]'
    foreach($k in @('job_id','scene_sha256','source_sha256','parameters_sha256')){$lines.Add($k+'='+$Submission.$k)}
    foreach($item in $parameters.PSObject.Properties){$lines.Add($item.Name+'='+[Convert]::ToString($item.Value,[Globalization.CultureInfo]::InvariantCulture))}
    [IO.File]::WriteAllLines((Join-Path $job 'case.properties'),$lines,$script:Utf8)
    foreach($name in @('UltraResearch.java','run-research.ps1')){Copy-Item -LiteralPath (Join-Path (Join-Path $Root 'research-template') $name) -Destination (Join-Path $job $name)}
    # Worker and helpers are copied too: future interface updates cannot change an active job.
    foreach($name in @('Run-ResearchJob.ps1','Research.Common.ps1','Http.Common.ps1')){Copy-Item -LiteralPath (Join-Path $Root $name) -Destination (Join-Path $job $name)}
    Write-JsonFile (Join-Path $job 'status.json') @{job_id=$Submission.job_id;kind=$Submission.kind;phase='starting';updated_at_utc=[DateTime]::UtcNow.ToString('o')}
    return [pscustomobject]@{Directory=$job;Created=$true}
}
function Get-WorkAILiveLog($Root,$Id,$Name) {
    Assert-JobId $Id
    if($Name -cnotin @('worker.log','solver.log','compile.log')){throw 'Only solver logs may be read live.'}
    $job=Join-Path (Join-Path $Root 'jobs') $Id;$state=Read-JsonFile (Join-Path $job 'status.json')
    $path=Join-Path $job $Name
    if(-not(Test-Path -LiteralPath $path -PathType Leaf)){return @{schema='workai.comsol.live_log.v1';job_id=$Id;phase=$state.phase;name=$Name;exists=$false;sampled_at_utc=[DateTime]::UtcNow.ToString('o')}}
    $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    try{
        $total=$stream.Length;$count=[int][Math]::Min(24576,$total);$null=$stream.Seek(-$count,[IO.SeekOrigin]::End)
        $buffer=New-Object byte[] $count;$offset=0
        while($offset -lt $count){$n=$stream.Read($buffer,$offset,$count-$offset);if($n -eq 0){break};$offset+=$n}
    }finally{$stream.Dispose()}
    # Preserve raw bytes; the cloud decodes COMSOL's UTF-8/Windows code page explicitly.
    $worker=$null;if($state.PSObject.Properties['worker_pid']){$worker=$state.worker_pid}
    return @{schema='workai.comsol.live_log.v1';job_id=$Id;phase=$state.phase;name=$Name;exists=$true;total_bytes=$total;tail_base64=[Convert]::ToBase64String($buffer,0,$offset);last_write_utc=(Get-Item -LiteralPath $path).LastWriteTimeUtc.ToString('o');sampled_at_utc=[DateTime]::UtcNow.ToString('o');worker_pid=$worker}
}
function Update-ResearchModel($Root,$Submission) {
    $keys=@('revision_id','expected_source_sha256','source_sha256','java_source')
    $actual=@($Submission.PSObject.Properties.Name)
    if($actual.Count -ne 4 -or @($actual | Where-Object {$_ -notin $keys}).Count){throw 'Unexpected revision fields.'}
    if($Submission.revision_id -cnotmatch '^research-[a-f0-9]{32}$' -or $Submission.source_sha256 -cnotmatch '^[a-f0-9]{64}$' -or $Submission.expected_source_sha256 -cnotmatch '^[a-f0-9]{64}$'){throw 'Invalid revision identity.'}
    $source=[string]$Submission.java_source
    if($source.Length -gt 45000 -or $source.Length -lt 100 -or $source.Contains([char]0) -or $source -cnotmatch 'public class UltraResearch\s*\{'){throw 'Expected bounded UltraResearch.java source.'}
    if((Get-BytesHash $script:Utf8.GetBytes($source)) -cne $Submission.source_sha256){throw 'Source digest differs.'}
    $manifest=Assert-ResearchTemplate $Root
    if($manifest.files.'research-template/UltraResearch.java' -ceq $Submission.source_sha256){return @{status='installed';source_sha256=$Submission.source_sha256}}
    if($manifest.files.'research-template/UltraResearch.java' -cne $Submission.expected_source_sha256){throw 'Installed research revision changed; refresh health.'}
    $revision=Join-Path (Join-Path $Root 'research-revisions') $Submission.revision_id
    if(Test-Path -LiteralPath $revision){throw 'Revision ID already used; inspect the previous result.'}
    $null=[IO.Directory]::CreateDirectory($revision)
    [IO.File]::WriteAllText((Join-Path $revision 'UltraResearch.java'),$source,$script:Utf8)
    Write-JsonFile (Join-Path $revision 'request.json') $Submission
    $settings=Read-JsonFile (Join-Path $Root 'settings.json')
    Push-Location -LiteralPath $revision
    try{$saved=$ErrorActionPreference;try{$ErrorActionPreference='Continue';& (Join-Path $settings.comsol_bin 'comsolcompile.exe') 'UltraResearch.java' *> 'compile.log';$code=$LASTEXITCODE}finally{$ErrorActionPreference=$saved}}finally{Pop-Location}
    if($code -ne 0 -or -not(Test-Path (Join-Path $revision 'UltraResearch.class'))){
        $failure=@{status='compile_failed';source_sha256=$Submission.source_sha256;compile_tail=@(Get-Content -LiteralPath (Join-Path $revision 'compile.log') -Tail 30)}
        Write-JsonFile (Join-Path $revision 'status.json') $failure;return $failure
    }
    $destination=Join-Path $Root 'research-template/UltraResearch.java';$manifestPath=Join-Path $Root 'research-manifest.json'
    Copy-Item -LiteralPath $destination -Destination (Join-Path $revision 'previous.java')
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $revision 'previous-manifest.json')
    try{
        [IO.File]::WriteAllText($destination,$source,$script:Utf8)
        $manifest.files.'research-template/UltraResearch.java'=$Submission.source_sha256;Write-JsonFile $manifestPath $manifest
        $null=Assert-ResearchTemplate $Root
    }catch{Copy-Item -LiteralPath (Join-Path $revision 'previous.java') -Destination $destination -Force;Copy-Item -LiteralPath (Join-Path $revision 'previous-manifest.json') -Destination $manifestPath -Force;throw}
    $result=@{status='installed';source_sha256=$Submission.source_sha256;revision_id=$Submission.revision_id;compiled=$true;model_run_proven=$false}
    Write-JsonFile (Join-Path $revision 'status.json') $result;return $result
}
