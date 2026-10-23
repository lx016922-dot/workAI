[CmdletBinding()]
param(
    [string]$InstallationFolder = 'C:\Users\Administrator\Desktop\workai-comsol-http-v1\workai-comsol-http-v1',
    [string]$OutputDirectory = [Environment]::GetFolderPath('Desktop')
)

# Read-only source export. Never dot-source or run anything from the installation.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-OrdinaryPath([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    while ($null -ne $item) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Export stopped: a selected path contains a link or junction.'
        }
        if ($item -is [IO.DirectoryInfo]) { $item = $item.Parent }
        else { $item = $item.Directory }
    }
}

function Get-BytesSha256([byte[]]$Bytes) {
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hasher.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $hasher.Dispose() }
}

function Assert-NoPrivateMaterial([byte[]]$Bytes) {
    $buffer = [IO.MemoryStream]::new($Bytes, $false)
    $reader = [IO.StreamReader]::new($buffer, [Text.Encoding]::UTF8, $true)
    try { $sourceText = $reader.ReadToEnd() }
    finally { $reader.Dispose(); $buffer.Dispose() }
    if ($sourceText -match '(?is)-----BEGIN [^-\r\n]*PRIVATE KEY-----' -or
        $sourceText -match '(?is)<RSAKeyValue>.*<(D|P|Q|DP|DQ|InverseQ)>' -or
        $sourceText -match '(?i)tskey-(auth|api)-[A-Za-z0-9_-]{8,}') {
        throw 'Export stopped: selected source contains possible private credentials. No archive was created.'
    }
}

if (-not (Test-Path -LiteralPath $InstallationFolder -PathType Container)) {
    throw 'Installation folder not found. Use -InstallationFolder with the original folder containing settings.json.'
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { throw 'Output directory is required.' }
Assert-OrdinaryPath $InstallationFolder
$sourceRoot = [IO.Path]::GetFullPath($InstallationFolder).TrimEnd([char[]]@([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar))
$destinationRoot = [IO.Path]::GetFullPath($OutputDirectory)
$sourcePrefix = $sourceRoot + [IO.Path]::DirectorySeparatorChar
if ($destinationRoot.Equals($sourceRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $destinationRoot.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Choose an output directory outside the installation folder.'
}
if (-not (Test-Path -LiteralPath $destinationRoot -PathType Container)) {
    throw 'Output directory does not exist.'
}
Assert-OrdinaryPath $destinationRoot

# Exact allowlist, never recurse or consume paths from an installed manifest.
$allowed = @(
    'research-template/UltraResearch.java',
    'Research.Common.ps1',
    'research-manifest.json',
    'Run-ResearchJob.ps1',
    'research-template/Run-ResearchJob.ps1'
)
$expectedSourceHash = 'ac090340bd65c47240b7954d2a51125b010f18d0e608bed012a6525febacccd7'
$files = [Collections.Generic.List[object]]::new()
$missing = [Collections.Generic.List[string]]::new()
foreach ($relative in $allowed) {
    $fullPath = Join-Path $sourceRoot $relative
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        $missing.Add($relative)
        continue
    }
    Assert-OrdinaryPath $fullPath
    if ((Get-Item -LiteralPath $fullPath).Length -gt 2097152) {
        throw 'A selected source file exceeds the 2 MiB export limit. No archive was created.'
    }
    $bytes = [IO.File]::ReadAllBytes($fullPath)
    if ($bytes.Length -gt 2097152) { throw 'Source changed size during reading. No archive was created.' }
    Assert-NoPrivateMaterial $bytes
    $files.Add([pscustomobject]@{ path = $relative; bytes = $bytes; sha256 = Get-BytesSha256 $bytes })
}
$model = @($files | Where-Object { $_.path -ceq 'research-template/UltraResearch.java' })
if ($model.Count -ne 1) {
    throw 'research-template/UltraResearch.java was not found. Check the original installation folder. No archive was created.'
}
$manifest = [ordered]@{
    schema = 'workai.research.source-export.v1'
    exported_at_utc = [DateTime]::UtcNow.ToString('o')
    source_files_executed = $false
    installation_files_modified = $false
    signed_health_checked_at_utc = '2026-10-22T05:29:47Z'
    signed_health_source_sha256 = $expectedSourceHash
    source_matches_observed_health = ($model[0].sha256 -ceq $expectedSourceHash)
    files = @($files | ForEach-Object { [ordered]@{ path = $_.path; bytes = $_.bytes.Length; sha256 = $_.sha256 } })
    missing_optional_files = @($missing.ToArray())
}
$utf8 = [Text.UTF8Encoding]::new($false)
$manifestBytes = $utf8.GetBytes(($manifest | ConvertTo-Json -Depth 8) + "`n")
$files.Add([pscustomobject]@{ path = 'export-manifest.json'; bytes = $manifestBytes; sha256 = Get-BytesSha256 $manifestBytes })

Add-Type -AssemblyName System.IO.Compression
$outputName = 'workai-research-source-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8) + '.zip'
$outputPath = Join-Path $destinationRoot $outputName
$stream = $null
$archive = $null
$created = $false
try {
    $stream = [IO.File]::Open($outputPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    $created = $true
    $archive = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create, $true)
    foreach ($file in $files) {
        $entry = $archive.CreateEntry($file.path, [IO.Compression.CompressionLevel]::Optimal)
        $entryStream = $entry.Open()
        try { $entryStream.Write($file.bytes, 0, $file.bytes.Length) }
        finally { $entryStream.Dispose() }
    }
    $archive.Dispose(); $archive = $null
    $stream.Dispose(); $stream = $null
} catch {
    if ($null -ne $archive) { $archive.Dispose() }
    if ($null -ne $stream) { $stream.Dispose() }
    if ($created -and (Test-Path -LiteralPath $outputPath -PathType Leaf)) { Remove-Item -LiteralPath $outputPath }
    throw
}
Write-Host 'SOURCE EXPORT COMPLETE. No installed files were changed.'
Write-Host ('ZIP: ' + $outputPath)
Write-Host ('Matches observed research source: ' + $manifest.source_matches_observed_health)
Write-Host 'Selected files:'
$manifest.files | ForEach-Object { Write-Host ('  ' + $_.path) }
if ($missing.Count -gt 0) { Write-Host ('Missing optional files: ' + ($missing -join ', ')) }
Write-Host 'Upload the new workai-research-source ZIP to the current conversation.'
