#Requires -Version 5.1
# Numerical jobs accept bounded data only. Model publication is a separate signed route.
function Assert-PlateKeys($Object,[string[]]$Keys,[string]$Label) {
    $actual=@($Object.PSObject.Properties.Name)
    if ($actual.Count -ne $Keys.Count -or @($actual | Where-Object {$_ -cnotin $Keys}).Count) { throw ($Label+': unexpected or missing fields.') }
}
function Assert-PlateNumber($Value,[double]$Min,[double]$Max,[string]$Label) {
    if ($Value -is [bool] -or $Value -is [string] -or $null -eq $Value) { throw ($Label+': number required.') }
    $v=[double]$Value
    if ([double]::IsNaN($v) -or [double]::IsInfinity($v) -or $v -lt $Min -or $v -gt $Max) { throw ($Label+': outside approved range.') }
}
function Assert-PlateCase($Case) {
    if($Case.schema -ceq 'workai.physics_ai.volume.case.v1'){Assert-VolumeCase $Case;return}
    Assert-PlateKeys $Case @('schema','geometry','material','boundary','notch','sources','receivers','acquisition','numerics') 'case'
    if ($Case.schema -cne 'workai.comsol.plate.case.v1') { throw 'Unknown plate case schema.' }
    Assert-PlateKeys $Case.geometry @('length_mm','width_mm','thickness_mm') 'geometry'
    Assert-PlateKeys $Case.material @('rho_kg_m3','cl_m_s','ct_m_s') 'material'
    Assert-PlateKeys $Case.notch @('enabled','x_mm','y_mm','length_mm','width_mm','depth_mm') 'notch'
    Assert-PlateKeys $Case.acquisition @('duration_us','sample_step_us') 'acquisition'
    Assert-PlateKeys $Case.numerics @('coarse_h_mm','fine_h_mm','max_step_us','relative_tolerance','field_nx','field_ny','field_frames') 'numerics'
    $g=$Case.geometry; $m=$Case.material; $n=$Case.numerics; $a=$Case.acquisition; $c=$Case.notch
    Assert-PlateNumber $g.length_mm 15 100 'length'; Assert-PlateNumber $g.width_mm 8 60 'width'; Assert-PlateNumber $g.thickness_mm 1 5 'thickness'
    Assert-PlateNumber $m.rho_kg_m3 500 20000 'density'; Assert-PlateNumber $m.cl_m_s 1500 10000 'cL'; Assert-PlateNumber $m.ct_m_s 500 6000 'cT'
    if ($m.cl_m_s*$m.cl_m_s -le 4*$m.ct_m_s*$m.ct_m_s/3) { throw 'Nonpositive bulk modulus.' }
    if ($Case.boundary -cnotin @('free','side_impedance')) { throw 'Unknown side boundary.' }
    if ($c.enabled -isnot [bool]) { throw 'notch.enabled must be boolean.' }
    foreach ($k in @('x_mm','y_mm','length_mm','width_mm','depth_mm')) { Assert-PlateNumber $c.$k 0.1 100 $k }
    if ($c.width_mm -lt 0.2 -or $c.width_mm -gt 2 -or $c.depth_mm -ge $g.thickness_mm -or
        $c.x_mm -le $c.width_mm/2 -or $c.x_mm -ge $g.length_mm-$c.width_mm/2 -or
        $c.y_mm -le $c.length_mm/2 -or $c.y_mm -ge $g.width_mm-$c.length_mm/2) { throw 'Use an interior partial-depth open surface notch.' }
    if ($Case.sources -isnot [array] -or @($Case.sources).Count -lt 1 -or @($Case.sources).Count -gt 3 -or
        $Case.receivers -isnot [array] -or @($Case.receivers).Count -lt 1 -or @($Case.receivers).Count -gt 8) { throw 'Use 1-3 sources and 1-8 receivers.' }
    $bounds=@($g.length_mm,$g.width_mm,$g.thickness_mm)
    foreach ($group in @('sources','receivers')) {
        $ids=@{}
        foreach ($point in $Case.$group) {
            $keys=@('id','position_mm'); if ($group -eq 'sources') {$keys+=@('direction','radius_mm','frequencies_mhz','cycles','delay_us','amplitude_pa','phase_deg')}
            Assert-PlateKeys $point $keys $group
            if ($point.id -cnotmatch '^[A-Za-z][A-Za-z0-9_-]{0,15}$' -or $ids.ContainsKey($point.id)) { throw 'Unique short IDs required.' }; $ids[$point.id]=$true
            if ($point.position_mm -isnot [array] -or $point.position_mm.Count -ne 3) { throw 'Three coordinates required.' }
            for ($i=0;$i -lt 3;$i++) {Assert-PlateNumber $point.position_mm[$i] 0 $bounds[$i] 'point'}
            if ([Math]::Abs($point.position_mm[2]-$g.thickness_mm) -gt 1e-8) {throw 'Initial contract requires top surface points.'}
            if ($c.enabled -and [Math]::Abs($point.position_mm[0]-$c.x_mm) -le $c.width_mm/2 -and [Math]::Abs($point.position_mm[1]-$c.y_mm) -le $c.length_mm/2) {throw 'Point is in the open notch.'}
        }
    }
    Assert-PlateNumber $a.duration_us 5 150 'duration'; Assert-PlateNumber $a.sample_step_us .05 .5 'sampling'
    if ([Math]::Abs($a.duration_us/$a.sample_step_us-[Math]::Round($a.duration_us/$a.sample_step_us)) -gt 1e-7) {throw 'Duration must be an integer sample count.'}
    foreach ($s in $Case.sources) {
        if ($s.direction -isnot [array] -or $s.direction.Count -ne 3) {throw 'Three direction components required.'}
        $norm=0.; foreach ($v in $s.direction) {Assert-PlateNumber $v -1 1 'direction';$norm+=$v*$v}
        if ([Math]::Abs($norm-1) -gt 1e-8) {throw 'Direction must be a unit vector.'}
        Assert-PlateNumber $s.radius_mm .5 5 'radius';Assert-PlateNumber $s.cycles 2 8 'cycles';Assert-PlateNumber $s.delay_us 0 100 'delay'
        Assert-PlateNumber $s.amplitude_pa 1 1e7 'traction';Assert-PlateNumber $s.phase_deg -360 360 'phase'
        if ($s.frequencies_mhz -isnot [array] -or $s.frequencies_mhz.Count -lt 1 -or $s.frequencies_mhz.Count -gt 3 -or @($s.frequencies_mhz | Select-Object -Unique).Count -ne $s.frequencies_mhz.Count) {throw 'Use 1-3 distinct frequencies.'}
        foreach ($freq in $s.frequencies_mhz) {Assert-PlateNumber $freq .08 .35 'frequency';if($s.delay_us+$s.cycles/$freq -ge $a.duration_us){throw 'Pulse exceeds acquisition time.'}}
    }
    Assert-PlateNumber $n.coarse_h_mm .8 3 'coarse mesh';Assert-PlateNumber $n.fine_h_mm .5 2 'fine mesh'
    Assert-PlateNumber $n.max_step_us .01 .25 'max step';Assert-PlateNumber $n.relative_tolerance 1e-7 1e-4 'tolerance'
    if ($n.fine_h_mm -ge $n.coarse_h_mm -or $n.max_step_us -gt $a.sample_step_us) {throw 'Mesh/time refinement required.'}
    foreach ($spec in @(@('field_nx',5,61),@('field_ny',5,41),@('field_frames',5,101))) {
        $v=$n.($spec[0]);Assert-PlateNumber $v $spec[1] $spec[2] $spec[0];if($v -ne [Math]::Round($v)){throw 'Integer field sampling required.'}
    }
    if ($g.length_mm*$g.width_mm*$g.thickness_mm/[Math]::Pow($n.fine_h_mm,3) -gt 25000) {throw 'Case exceeds initial 3-D mesh budget.'}
}
function Assert-PlateTemplate([string]$Root) {
    $manifest=Read-JsonFile (Join-Path $Root 'plate-manifest.json')
    if ($manifest.schema -cne 'workai.comsol.plate.bundle.v1') {throw 'Unknown plate bundle.'}
    $names=@('UltraPlate.java','run-plate.ps1','Plate.Common.ps1','Run-PlateJob.ps1')
    if($manifest.files.PSObject.Properties['Volume.Protocol.ps1']){$names+=@('Volume.Protocol.ps1')}
    Assert-PlateKeys $manifest.files $names 'plate manifest'
    foreach ($name in $names) {
        $base=if($name -in @('UltraPlate.java','run-plate.ps1')){Join-Path $Root 'plate-template'}else{$Root}
        if ((Get-FileHashText (Join-Path $base $name)) -cne $manifest.files.$name) {throw ('Plate template changed: '+$name)}
    }
    return $manifest
}
function New-PlateJobFiles([string]$Root,$Submission) {
    Assert-PlateKeys $Submission @('job_id','kind','prepared_at_utc','case_sha256','source_sha256','case_json') 'submission'
    Assert-JobId $Submission.job_id
    if ($Submission.kind -cnotin @('plate_smoke','plate_reference')) {throw 'Unknown plate task kind.'}
    if ($Submission.case_json -isnot [string] -or $Submission.case_json.Length -gt 30000) {throw 'Invalid plate case document.'}
    if ($Submission.prepared_at_utc -notmatch '(Z|\+00:00)$') {throw 'UTC time required.'}
    $prepared=[DateTimeOffset]::Parse($Submission.prepared_at_utc,[Globalization.CultureInfo]::InvariantCulture)
    if ($prepared -gt [DateTimeOffset]::UtcNow.AddMinutes(5) -or $prepared -lt [DateTimeOffset]::UtcNow.AddDays(-1)) {throw 'Prepare a fresh task.'}
    $manifest=Assert-PlateTemplate $Root
    if ($Submission.source_sha256 -cne $manifest.files.'UltraPlate.java') {throw 'Installed plate model differs from requested model.'}
    if ((Get-BytesHash ($script:Utf8.GetBytes($Submission.case_json))) -cne $Submission.case_sha256) {throw 'Plate case digest mismatch.'}
    $case=ConvertFrom-WorkAIJson $Submission.case_json;Assert-PlateCase $case
    if($case.schema -ceq 'workai.physics_ai.volume.case.v1' -and -not ([IO.File]::ReadAllText((Join-Path $Root 'plate-template/UltraPlate.java'))).Contains('WORKAI_VOLUME_DISPLACEMENT_V1')){throw 'The input protocol is installed; a compatible signed large-volume model is still required.'}
    $jobs=Join-Path $Root 'jobs';$job=Join-Path $jobs $Submission.job_id
    if (Test-Path -LiteralPath $job) {
        $old=Read-JsonFile (Join-Path $job 'submission.json')
        foreach ($key in @('job_id','kind','prepared_at_utc','case_sha256','source_sha256','case_json')) {if($old.$key -cne $Submission.$key){throw 'Job ID belongs to another request.'}}
        return [pscustomobject]@{Directory=$job;Created=$false}
    }
    foreach ($d in @(Get-ChildItem -LiteralPath $jobs -Directory)) {
        $state=Read-JsonFile (Join-Path $d.FullName 'status.json')
        if($state.phase -notin @('completed','failed')){throw 'An active or unresolved job exists. Do not resubmit.'}
    }
    $null=[IO.Directory]::CreateDirectory($job)
    Write-JsonFile (Join-Path $job 'submission.json') $Submission
    Write-JsonFile (Join-Path $job 'request.json') ([ordered]@{schema='workai.comsol.plate.request.v1';job_id=$Submission.job_id;kind=$Submission.kind;case=$case;case_json=$Submission.case_json;case_sha256=$Submission.case_sha256;source_sha256=$Submission.source_sha256;prepared_at_utc=$Submission.prepared_at_utc})
    if($case.schema -ceq 'workai.physics_ai.volume.case.v1'){$props=Get-VolumeProperties $case $Submission}else{
    $props=[ordered]@{schema=$case.schema;job_id=$Submission.job_id;kind=$Submission.kind;case_sha256=$Submission.case_sha256;source_sha256=$Submission.source_sha256;boundary=$case.boundary;source_count=@($case.sources).Count;receiver_count=@($case.receivers).Count}
    foreach($section in @('geometry','material','notch','acquisition','numerics')){foreach($entry in $case.$section.PSObject.Properties){$v=$entry.Value;if($v -is [bool]){$v=[int]$v};$props[$section+'_'+$entry.Name]=$v}}
    foreach($group in @('sources','receivers')){
        $prefix=if($group -eq 'sources'){'tx'}else{'rx'};$i=0
        foreach($point in $case.$group){
            $stem=$prefix+$i+'_';$props[$stem+'id']=$point.id
            for($j=0;$j -lt 3;$j++){$props[$stem+'p'+$j]=$point.position_mm[$j]}
            if($group -eq 'sources'){
                for($j=0;$j -lt 3;$j++){$props[$stem+'d'+$j]=$point.direction[$j]}
                $props[$stem+'frequency_count']=$point.frequencies_mhz.Count
                for($j=0;$j -lt $point.frequencies_mhz.Count;$j++){$props[$stem+'f'+$j]=$point.frequencies_mhz[$j]}
                foreach($key in @('radius_mm','cycles','delay_us','amplitude_pa','phase_deg')){$props[$stem+$key]=$point.$key}
            };$i++
        }
    }
    }
    $lines=@();foreach($key in $props.Keys){$lines+=($key+'='+[Convert]::ToString($props[$key],[Globalization.CultureInfo]::InvariantCulture))}
    [IO.File]::WriteAllText((Join-Path $job 'case.properties'),(($lines -join "`n")+"`n"),$script:Utf8)
    foreach($name in @('UltraPlate.java','run-plate.ps1')){Copy-Item -LiteralPath (Join-Path (Join-Path $Root 'plate-template') $name) -Destination (Join-Path $job $name)}
    Write-JsonFile (Join-Path $job 'status.json') @{job_id=$Submission.job_id;kind=$Submission.kind;phase='starting';updated_at_utc=[DateTime]::UtcNow.ToString('o')}
    return [pscustomobject]@{Directory=$job;Created=$true}
}

# Only the enrolled, signature-authenticated cloud client may publish a model.
# Java is executable model code, not data; this is deliberately a separate route.
# Destination, compiler, class name and runner stay fixed. No shell or path input.
function Update-PlateModel([string]$Root,$Submission) {
    Assert-PlateKeys $Submission @('revision_id','expected_source_sha256','source_sha256','java_source') 'model revision'
    if($Submission.revision_id -cnotmatch '^plate-[a-f0-9]{32}$'){throw 'Invalid model revision ID.'}
    foreach($key in @('source_sha256','expected_source_sha256')){if($Submission.$key -cnotmatch '^[a-f0-9]{64}$'){throw 'Invalid model source hash.'}}
    if($Submission.java_source -isnot [string] -or $Submission.java_source.Length -gt 45000 -or $Submission.java_source.Contains([char]0) -or $Submission.java_source -cnotmatch 'public\s+class\s+UltraPlate\b'){throw 'Provide only the UltraPlate Java model source (maximum 45000 characters).'}
    $sourceBytes=$script:Utf8.GetBytes($Submission.java_source)
    if((Get-BytesHash $sourceBytes) -cne $Submission.source_sha256){throw 'Model source digest mismatch.'}
    $manifest=Assert-PlateTemplate $Root
    $revision=Join-Path (Join-Path $Root 'model-revisions') $Submission.revision_id
    if(Test-Path -LiteralPath $revision){
        $old=Read-JsonFile (Join-Path $revision 'submission.json')
        foreach($key in @('revision_id','expected_source_sha256','source_sha256','java_source')){if($old.$key -cne $Submission.$key){throw 'Revision ID belongs to another request.'}}
        return (Read-JsonFile (Join-Path $revision 'status.json'))
    }
    if($manifest.files.'UltraPlate.java' -cne $Submission.expected_source_sha256){throw 'Installed source changed; inspect current revision before updating.'}
    foreach($job in @(Get-ChildItem -LiteralPath (Join-Path $Root 'jobs') -Directory)){
        $state=Read-JsonFile (Join-Path $job.FullName 'status.json')
        if($state.phase -notin @('completed','failed')){throw 'An active or unresolved job prevents a model update.'}
    }
    $null=[IO.Directory]::CreateDirectory($revision)
    Write-JsonFile (Join-Path $revision 'submission.json') $Submission
    $state=[ordered]@{schema='workai.comsol.plate.revision.v1';revision_id=$Submission.revision_id;source_sha256=$Submission.source_sha256;previous_source_sha256=$manifest.files.'UltraPlate.java';phase='compiling';created_at_utc=[DateTime]::UtcNow.ToString('o');model_solved=$false}
    Write-JsonFile (Join-Path $revision 'status.json') $state
    [IO.File]::WriteAllBytes((Join-Path $revision 'UltraPlate.java'),$sourceBytes)
    $oldFiles=@{};$changed=$false
    try{
        $settings=Read-JsonFile (Join-Path $Root 'settings.json')
        Push-Location -LiteralPath $revision
        try{
            $saved=$ErrorActionPreference
            try{$ErrorActionPreference='Continue';& (Join-Path $settings.comsol_bin 'comsolcompile.exe') 'UltraPlate.java' *> 'compile.log';$code=$LASTEXITCODE}finally{$ErrorActionPreference=$saved}
            if($code -ne 0 -or -not(Test-Path 'UltraPlate.class')){throw 'COMSOL model compilation failed; previous model is still installed.'}
        }finally{Pop-Location}
        foreach($name in @('plate-template/UltraPlate.java','plate-manifest.json','public/http-connection-info.json')){
            $oldFiles[$name]=[IO.File]::ReadAllBytes((Join-Path $Root $name))
            $backup=Join-Path (Join-Path $revision 'previous') $name
            $null=[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($backup));[IO.File]::WriteAllBytes($backup,$oldFiles[$name])
        }
        $changed=$true
        [IO.File]::WriteAllBytes((Join-Path $Root 'plate-template/UltraPlate.java'),$sourceBytes)
        $manifest.files.'UltraPlate.java'=$Submission.source_sha256
        Write-JsonFile (Join-Path $Root 'plate-manifest.json') $manifest
        $report=Read-JsonFile (Join-Path $Root 'public/http-connection-info.json')
        $report | Add-Member -NotePropertyName plate_source_sha256 -NotePropertyValue $Submission.source_sha256 -Force
        $report | Add-Member -NotePropertyName plate_model_tested -NotePropertyValue $false -Force
        Write-JsonFile (Join-Path $Root 'public/http-connection-info.json') $report
        $null=Assert-PlateTemplate $Root
        $state.phase='applied';$state['updated_at_utc']=[DateTime]::UtcNow.ToString('o')
        Write-JsonFile (Join-Path $revision 'status.json') $state
    }catch{
        $failure=$_.Exception.Message
        if($changed){foreach($name in $oldFiles.Keys){[IO.File]::WriteAllBytes((Join-Path $Root $name),$oldFiles[$name])}}
        $state.phase='failed';$state['error']=$failure;$state['updated_at_utc']=[DateTime]::UtcNow.ToString('o')
        if(Test-Path (Join-Path $revision 'compile.log')){$state['compiler_tail']=(@(Get-Content -LiteralPath (Join-Path $revision 'compile.log') -Tail 30) -join "`n")}
        Write-JsonFile (Join-Path $revision 'status.json') $state
    }
    return $state
}

if(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'Volume.Protocol.ps1')){. (Join-Path $PSScriptRoot 'Volume.Protocol.ps1')}
