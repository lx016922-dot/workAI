#Requires -Version 5.1
[CmdletBinding()]
param([string]$Root='',[switch]$NoStart,[switch]$PromptForRoot)
$ErrorActionPreference='Stop'
$old=@{};$changed=$false
$stage='locate the update and existing installation'

function Test-VolumeInstallation([string]$Directory) {
    if([string]::IsNullOrWhiteSpace($Directory)){return $false}
    foreach($name in @('settings.json','Http.Common.ps1','Plate.Common.ps1')){
        if(-not [IO.File]::Exists([IO.Path]::Combine($Directory,$name))){return $false}
    }
    return $true
}

function Find-VolumeInstallation([string]$UpdateDirectory,[string]$DesktopDirectory) {
    # An update copied into an installation explicitly selects that installation.
    if(Test-VolumeInstallation $UpdateDirectory){return $UpdateDirectory}
    $parent=[IO.Path]::GetDirectoryName($UpdateDirectory)
    $candidates=@($parent)
    foreach($base in @($parent,$DesktopDirectory)){
        if(-not [string]::IsNullOrWhiteSpace($base)){
            $container=Join-Path $base 'workai-comsol-http-v1'
            $candidates+=@($container,(Join-Path $container 'workai-comsol-http-v1'))
        }
    }
    $found=@()
    foreach($candidate in $candidates){
        if(Test-VolumeInstallation $candidate){
            $full=(Get-Item -LiteralPath $candidate -Force).FullName
            if($found -notcontains $full){$found+=@($full)}
        }
    }
    return $found
}

function Resolve-VolumeInstallation([string]$Directory) {
    if([string]::IsNullOrWhiteSpace($Directory)){throw 'No installation folder was supplied. Use the existing folder containing settings.json.'}
    $Directory=$Directory.Trim()
    # Explorer's Copy as path includes quotes; accept one matching pair.
    if($Directory.Length -ge 2 -and $Directory.StartsWith('"') -and $Directory.EndsWith('"')){$Directory=$Directory.Substring(1,$Directory.Length-2)}
    if([string]::IsNullOrWhiteSpace($Directory) -or -not [IO.Path]::IsPathRooted($Directory)){throw 'Use the full path to the existing folder containing settings.json.'}
    try{$item=Get-Item -LiteralPath $Directory -Force -ErrorAction Stop}catch{throw ('Installation folder was not found: '+$Directory)}
    if($item -isnot [IO.DirectoryInfo] -or -not (Test-VolumeInstallation $item.FullName)){throw ('This is not the existing COMSOL API installation: '+$Directory+'. Select the folder containing settings.json, Http.Common.ps1 and Plate.Common.ps1. Do not run Setup again.')}
    return $item.FullName
}

if($MyInvocation.InvocationName -eq '.'){return}
try {
    # Resolve the script location in the body, not a parameter default or cwd.
    $entry=$MyInvocation.MyCommand.Path
    if([string]::IsNullOrWhiteSpace($entry)){throw 'Extract the complete update ZIP, then run Update-Volume.cmd.'}
    $here=[IO.Path]::GetDirectoryName($entry)
    $payload=Join-Path $here 'volume-protocol-payload'
    if(-not [IO.File]::Exists((Join-Path $payload 'manifest.json'))){throw 'The volume-protocol-payload folder is missing. Extract the complete update ZIP before running it.'}
    if([string]::IsNullOrWhiteSpace($Root)){
        $found=@(Find-VolumeInstallation $here ([Environment]::GetFolderPath('Desktop')))
        if($found.Count -eq 1){$Root=$found[0]}
        elseif($PromptForRoot){
            if($found.Count -gt 1){Write-Host ('More than one installation was found: '+($found -join ' | '))}
            else{Write-Host 'The existing installation was not found automatically.'}
            Write-Host 'Paste the full path of the existing folder containing settings.json, then press Enter. Do not run Setup again.'
            $Root=Read-Host 'Installation folder'
        }else{
            throw ('Cannot locate one installation (found '+$found.Count+'). Run Update-Volume.cmd to select it, or pass -Root with the full path of the folder containing settings.json. No files were changed.')
        }
    }
    $Root=Resolve-VolumeInstallation $Root
    Write-Host ('Installation: '+$Root)
    $stage='check the existing settings and stopped API listener'
    . (Join-Path $Root 'Http.Common.ps1')
    . (Join-Path $Root 'Plate.Common.ps1')
    $settings=Read-JsonFile (Join-Path $Root 'settings.json')
    if(-not $settings.PSObject.Properties['schema'] -or -not $settings.PSObject.Properties['configured'] -or -not $settings.PSObject.Properties['base_url'] -or $settings.schema -cne 'workai.comsol.http.settings.v1' -or $settings.configured -ne $true){throw 'Existing HTTP setup is incomplete. No files were changed; do not run Setup again.'}
    if(Get-NetTCPConnection -State Listen -LocalPort ([Uri]$settings.base_url).Port -ErrorAction SilentlyContinue){throw 'Close only the API LISTENING window before updating.'}
    $stage='verify existing files, jobs and update payload'
    foreach($job in @(Get-ChildItem -LiteralPath (Join-Path $Root 'jobs') -Directory)){
        $state=Read-JsonFile (Join-Path $job.FullName 'status.json')
        if($state.phase -notin @('completed','failed')){throw 'A COMSOL job is active or unresolved; preserve it and wait before updating.'}
    }
    $plate=Assert-PlateTemplate $Root
    $bundle=Read-JsonFile (Join-Path $Root 'bundle-manifest.json')
    Assert-Template (Join-Path $Root 'template') $bundle
    $manifest=Read-JsonFile (Join-Path $payload 'manifest.json')
    if($manifest.schema -cne 'workai.comsol.volume.protocol.update.v1'){throw 'Invalid volume protocol update.'}
    $names=@('Plate.Common.ps1','Serve-Comsol.ps1','Volume.Protocol.ps1')
    foreach($name in @('Plate.Common.ps1','Serve-Comsol.ps1')){
        if((Get-FileHashText (Join-Path $Root $name)) -cne $manifest.expected_sha256.$name){throw ('Installed '+$name+' differs from the reviewed base. No files were changed.')}
    }
    foreach($name in $names){if((Get-FileHashText (Join-Path $payload $name)) -cne $manifest.payload_sha256.$name){throw ('Payload checksum failed: '+$name)}}
    # Parse every PowerShell file before changing the installation.
    foreach($name in $names){
        $tokens=$null;$errors=$null
        $null=[Management.Automation.Language.Parser]::ParseFile((Join-Path $payload $name),[ref]$tokens,[ref]$errors)
        if($errors.Count){throw ('PowerShell syntax check failed: '+$name+' '+($errors.Message -join ' | '))}
    }
    $stage='back up and install verified files'
    $backup=Join-Path (Join-Path $Root 'volume-protocol-backups') ([DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')+'-'+[Guid]::NewGuid().ToString('N'))
    $null=[IO.Directory]::CreateDirectory($backup)
    foreach($name in $names+@('plate-manifest.json','bundle-manifest.json')){
        $path=Join-Path $Root $name;$old[$name]=if(Test-Path -LiteralPath $path){[IO.File]::ReadAllBytes($path)}else{$null}
        if($null -ne $old[$name]){[IO.File]::WriteAllBytes((Join-Path $backup $name),$old[$name])}
    }
    $plate.files.'Plate.Common.ps1'=$manifest.payload_sha256.'Plate.Common.ps1'
    $plate.files | Add-Member -NotePropertyName 'Volume.Protocol.ps1' -NotePropertyValue $manifest.payload_sha256.'Volume.Protocol.ps1' -Force
    $bundle.files.'Serve-Comsol.ps1'=$manifest.payload_sha256.'Serve-Comsol.ps1'
    $changed=$true
    foreach($name in $names){[IO.File]::WriteAllBytes((Join-Path $Root $name),[IO.File]::ReadAllBytes((Join-Path $payload $name)))}
    Write-JsonFile (Join-Path $Root 'plate-manifest.json') $plate
    Write-JsonFile (Join-Path $Root 'bundle-manifest.json') $bundle
    . (Join-Path $Root 'Plate.Common.ps1')
    $null=Assert-PlateTemplate $Root
    Assert-Template (Join-Path $Root 'template') (Read-JsonFile (Join-Path $Root 'bundle-manifest.json'))
    $changed=$false
    Write-Host 'VOLUME PROTOCOL UPDATE COMPLETE' -ForegroundColor Green
    Write-Host 'Existing model source, keys, settings and jobs were preserved.'
    Write-Host 'Large-plate displacement jobs remain blocked until a compatible signed COMSOL model is installed and validated.'
    if(-not $NoStart){$stage='start the updated API';Write-Host 'Starting the API. Keep the API LISTENING window open.';& (Join-Path $Root 'Serve-Comsol.ps1') -Root $Root}
} catch {
    if($changed){foreach($name in $old.Keys){$path=Join-Path $Root $name;if($null -eq $old[$name]){if(Test-Path -LiteralPath $path){Remove-Item -LiteralPath $path}}else{[IO.File]::WriteAllBytes($path,$old[$name])}}}
    Write-Host ('Volume protocol update stopped: '+$_.Exception.Message) -ForegroundColor Red
    Write-Host ('Stage: '+$stage)
    Write-Host $_.InvocationInfo.PositionMessage
    exit 1
}
