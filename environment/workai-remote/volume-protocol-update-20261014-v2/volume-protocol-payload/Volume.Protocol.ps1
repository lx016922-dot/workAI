#Requires -Version 5.1
# Additive data protocol for the requested plate. No model or shell execution.
function Assert-VolumeCase($Case) {
    Assert-PlateKeys $Case @('schema','geometry','material','boundary','notch','sources','receivers','acquisition') 'volume case'
    if($Case.schema -cne 'workai.physics_ai.volume.case.v1'){throw 'Unknown volume schema.'}
    Assert-PlateKeys $Case.geometry @('length_mm','width_mm','thickness_mm') 'volume geometry'
    foreach($pair in @(@('length_mm',200),@('width_mm',100),@('thickness_mm',2))){Assert-PlateNumber $Case.geometry.($pair[0]) $pair[1] $pair[1] $pair[0]}
    Assert-PlateKeys $Case.material @('rho_kg_m3','cl_m_s','ct_m_s') 'volume material'
    foreach($pair in @(@('rho_kg_m3',2700),@('cl_m_s',6320),@('ct_m_s',3130))){Assert-PlateNumber $Case.material.($pair[0]) $pair[1] $pair[1] $pair[0]}
    Assert-PlateKeys $Case.boundary @('x_min','x_max','y_min','y_max','z_min','z_max') 'volume boundaries'
    foreach($face in @('x_min','x_max')){if($Case.boundary.$face -cnotin @('free','absorbing')){throw 'Unsupported end boundary.'}}
    foreach($face in @('y_min','y_max','z_min','z_max')){if($Case.boundary.$face -cne 'free'){throw 'This teacher requires free Y/Z surfaces.'}}
    Assert-PlateKeys $Case.notch @('enabled','x_mm','y_mm','length_mm','width_mm','depth_mm') 'notch'
    if($Case.notch.enabled -isnot [bool] -or $Case.notch.enabled){throw 'The large-volume teacher starts with an intact plate.'}
    foreach($key in @('x_mm','y_mm','length_mm','width_mm','depth_mm')){Assert-PlateNumber $Case.notch.$key .001 2000 $key}
    if($Case.sources -isnot [array] -or $Case.sources.Count -ne 1 -or $Case.receivers -isnot [array] -or $Case.receivers.Count -lt 1 -or $Case.receivers.Count -gt 8){throw 'Use one source and one to eight receivers.'}
    $source=$Case.sources[0]
    Assert-PlateKeys $source @('id','kind','position_mm','direction','width_mm','span_mm','amplitude_nm','frequencies_mhz','cycles','delay_us','phase_deg') 'volume displacement source'
    if($source.kind -cne 'prescribed_surface_displacement'){throw 'An explicit prescribed displacement source is required.'}
    foreach($group in @('sources','receivers')){
        $ids=@{}
        foreach($point in $Case.$group){
            if($group -eq 'receivers'){Assert-PlateKeys $point @('id','position_mm') 'volume receiver'}
            if($point.id -cnotmatch '^[A-Za-z][A-Za-z0-9_-]{0,15}$' -or $ids.ContainsKey($point.id)){throw 'Unique point IDs required.'};$ids[$point.id]=$true
            if($point.position_mm -isnot [array] -or $point.position_mm.Count -ne 3){throw 'XYZ coordinates required.'}
            for($j=0;$j -lt 3;$j++){Assert-PlateNumber $point.position_mm[$j] 0 (@(200,100,2)[$j]) 'coordinate'}
        }
    }
    if($source.direction -isnot [array] -or $source.direction.Count -ne 3){throw 'Three numeric direction components required.'}
    for($j=0;$j -lt 3;$j++){Assert-PlateNumber $source.direction[$j] (@(0,0,1)[$j]) (@(0,0,1)[$j]) 'normal direction'}
    if(($source.position_mm -join ',') -cne '10,50,2'){throw 'This teacher uses the normal source at (10,50,2) mm.'}
    Assert-PlateNumber $source.width_mm 1 1 'source X width';Assert-PlateNumber $source.span_mm 1 1 'source Y width'
    Assert-PlateNumber $source.amplitude_nm .1 10 'displacement nm';Assert-PlateNumber $source.cycles 3 7 'Hann cycles'
    Assert-PlateNumber $source.delay_us 0 10 'delay';Assert-PlateNumber $source.phase_deg -180 180 'phase'
    if($source.frequencies_mhz -isnot [array] -or $source.frequencies_mhz.Count -ne 1){throw 'One carrier frequency required.'}
    Assert-PlateNumber $source.frequencies_mhz[0] .2 .2 'carrier MHz'
    Assert-PlateKeys $Case.acquisition @('duration_us','sample_step_us') 'volume acquisition'
    Assert-PlateNumber $Case.acquisition.duration_us 150 150 'duration';Assert-PlateNumber $Case.acquisition.sample_step_us .2 1 'sample step'
    $intervals=$Case.acquisition.duration_us/$Case.acquisition.sample_step_us
    if([Math]::Abs($intervals-[Math]::Round($intervals)) -gt 1e-7){throw 'Sampling must divide the record.'}
}
function Get-VolumeProperties($Case,$Submission) {
    Assert-VolumeCase $Case
    $props=[ordered]@{schema=$Case.schema;job_id=$Submission.job_id;kind=$Submission.kind;case_sha256=$Submission.case_sha256;source_sha256=$Submission.source_sha256;boundary='faces';source_count=1;receiver_count=$Case.receivers.Count}
    foreach($section in @('geometry','material','notch','acquisition')){foreach($entry in $Case.$section.PSObject.Properties){$value=$entry.Value;if($value -is [bool]){$value=[int]$value};$props[$section+'_'+$entry.Name]=$value}}
    foreach($face in @('x_min','x_max','y_min','y_max','z_min','z_max')){$props['boundary_'+$face]=$Case.boundary.$face}
    $numerics=[ordered]@{coarse_h_mm=4.;fine_h_mm=3.;max_step_us=.05;relative_tolerance=1e-6;field_nx=101;field_ny=41;field_nz=3;field_frames=151;element_budget=30000}
    foreach($key in $numerics.Keys){$props['numerics_'+$key]=$numerics[$key]}
    foreach($group in @('sources','receivers')){
        $prefix=if($group -eq 'sources'){'tx'}else{'rx'};$i=0
        foreach($point in $Case.$group){
            $stem=$prefix+$i+'_';$props[$stem+'id']=$point.id
            for($j=0;$j -lt 3;$j++){$props[$stem+'p'+$j]=$point.position_mm[$j]}
            if($group -eq 'sources'){
                for($j=0;$j -lt 3;$j++){$props[$stem+'d'+$j]=$point.direction[$j]}
                foreach($key in @('kind','width_mm','span_mm','amplitude_nm','cycles','delay_us','phase_deg')){$props[$stem+$key]=$point.$key}
                $props[$stem+'frequency_count']=1;$props[$stem+'f0']=$point.frequencies_mhz[0]
            };$i++
        }
    }
    return $props
}
