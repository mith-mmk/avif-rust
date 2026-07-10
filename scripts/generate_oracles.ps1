[CmdletBinding()]
param(
    [string]$SourceAvif = "",
    [string]$OutputDir = "",
    [string]$FixtureId = "WML2Viewer",
    [string]$FfmpegPath = "ffmpeg",
    [string]$FfprobePath = "ffprobe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-RawVideoExport {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PixelFormat,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    & $FfmpegPath -v error -i $SourceAvif -frames:v 1 -f rawvideo -pix_fmt $PixelFormat -y $Destination 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg failed while exporting $PixelFormat"
    }
}

function Write-U16FromByteSamples {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]]$Samples,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    $output = [byte[]]::new($Samples.Length * 2)
    for ($index = 0; $index -lt $Samples.Length; $index++) {
        $output[$index * 2] = $Samples[$index]
    }
    [System.IO.File]::WriteAllBytes($Destination, $output)
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string[]]$Lines
    )

    [System.IO.File]::WriteAllLines(
        $Path,
        $Lines,
        [System.Text.UTF8Encoding]::new($false)
    )
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($SourceAvif)) {
    $SourceAvif = Join-Path (Split-Path -Parent $scriptDir) "..\samples\WML2Viewer.avif"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Split-Path -Parent $scriptDir) "test_data"
}

$sourcePath = (Resolve-Path -LiteralPath $SourceAvif).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputDir)
if ($FixtureId -notmatch '^[A-Za-z0-9._-]+$') {
    throw "FixtureId must contain only letters, digits, '.', '_' and '-'."
}

$probe = (& $FfprobePath -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x $sourcePath 2>&1) -join ""
if ($LASTEXITCODE -ne 0 -or $probe -notmatch '^([0-9]+)x([0-9]+)$') {
    throw "ffprobe did not return a valid video size for the AVIF source."
}
$width = [int]$Matches[1]
$height = [int]$Matches[2]
$planeSampleCount64 = [int64]$width * [int64]$height
if ($planeSampleCount64 -gt [int32]::MaxValue) {
    throw "The fixture dimensions exceed the supported sample-count limit."
}
$planeSampleCount = [int]$planeSampleCount64

$temporaryParent = Split-Path -Parent (Split-Path -Parent $outputPath)
$temporaryPath = Join-Path $temporaryParent ".test-avif-oracle"
$imagesPath = Join-Path $outputPath "images"
$planesPath = Join-Path $outputPath "planes"
$rgbaPath = Join-Path $outputPath "rgba"
New-Item -ItemType Directory -Force -Path $temporaryPath, $imagesPath, $planesPath, $rgbaPath | Out-Null

try {
    $gbrpTemporary = Join-Path $temporaryPath "$FixtureId.gbrp"
    $rgbaTemporary = Join-Path $temporaryPath "$FixtureId.rgba"
    $rgba16Temporary = Join-Path $temporaryPath "$FixtureId.rgba64le"
    Invoke-RawVideoExport -PixelFormat "gbrp" -Destination $gbrpTemporary
    Invoke-RawVideoExport -PixelFormat "rgba" -Destination $rgbaTemporary
    Invoke-RawVideoExport -PixelFormat "rgba64le" -Destination $rgba16Temporary

    $gbrp = [System.IO.File]::ReadAllBytes($gbrpTemporary)
    $expectedGbrpLength = $planeSampleCount * 3
    if ($gbrp.Length -ne $expectedGbrpLength) {
        throw "gbrp output length $($gbrp.Length) does not match $expectedGbrpLength."
    }
    $rgba = [System.IO.File]::ReadAllBytes($rgbaTemporary)
    $expectedRgbaLength = $planeSampleCount * 4
    if ($rgba.Length -ne $expectedRgbaLength) {
        throw "rgba output length $($rgba.Length) does not match $expectedRgbaLength."
    }
    $rgba16 = [System.IO.File]::ReadAllBytes($rgba16Temporary)
    $expectedRgba16Length = $planeSampleCount * 8
    if ($rgba16.Length -ne $expectedRgba16Length) {
        throw "rgba64le output length $($rgba16.Length) does not match $expectedRgba16Length."
    }

    $imageRelative = "images/$FixtureId.avif"
    $planeRelative = @(
        "planes/$FixtureId.y.u16le",
        "planes/$FixtureId.u.u16le",
        "planes/$FixtureId.v.u16le"
    )
    $rgbaRelative = "rgba/$FixtureId.rgba"
    $rgba16Relative = "rgba/$FixtureId.rgba16le"

    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $outputPath $imageRelative) -Force
    for ($plane = 0; $plane -lt 3; $plane++) {
        $planeBytes = $gbrp[($plane * $planeSampleCount)..(($plane + 1) * $planeSampleCount - 1)]
        Write-U16FromByteSamples -Samples $planeBytes -Destination (Join-Path $outputPath $planeRelative[$plane])
    }
    [System.IO.File]::WriteAllBytes((Join-Path $outputPath $rgbaRelative), $rgba)
    [System.IO.File]::WriteAllBytes((Join-Path $outputPath $rgba16Relative), $rgba16)

    $manifestPath = Join-Path $outputPath "oracles.csv"
    $manifestHeader = "id,avif,width,height,bit_depth,plane_paths,plane_widths,plane_heights,rgba8,rgba16"
    $manifestLine = "$FixtureId,$imageRelative,$width,$height,8,$($planeRelative -join ';'),$width;$width;$width,$height;$height;$height,$rgbaRelative,$rgba16Relative"
    $existingManifestLines = @()
    if (Test-Path -LiteralPath $manifestPath) {
        $existingManifestLines = @(Get-Content -LiteralPath $manifestPath | Where-Object {
            $_ -and $_ -ne $manifestHeader -and $_ -notmatch "^$([regex]::Escape($FixtureId)),"
        })
    }
    $manifest = @($manifestHeader) + $existingManifestLines + $manifestLine
    Write-Utf8NoBom -Path $manifestPath -Lines $manifest

    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash.ToLowerInvariant()
    $sourceManifestPath = Join-Path $outputPath "oracles.sources.csv"
    $sourceManifestHeader = "id,source,sha256,plane_format,generated_by"
    $existingSourceLines = @()
    if (Test-Path -LiteralPath $sourceManifestPath) {
        $existingSourceLines = @(Get-Content -LiteralPath $sourceManifestPath | Where-Object {
            $_ -and $_ -ne $sourceManifestHeader -and $_ -notmatch "^$([regex]::Escape($FixtureId)),"
        })
    }
    Write-Utf8NoBom -Path $sourceManifestPath -Lines @(
        $sourceManifestHeader,
        $existingSourceLines,
        "$FixtureId,$([System.IO.Path]::GetFileName($sourcePath)),$sourceHash,gbrp,generate_oracles.ps1"
    )

    Write-Host "generated oracle fixture $FixtureId ($width x $height, 8-bit gbrp)"
}
finally {
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Recurse -Force
    }
}
