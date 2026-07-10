[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$FixtureId = "filter-disabled-gbr",
    [ValidateSet("Gray", "Black", "VerticalSplit", "TestPattern", "Palette")]
    [string]$Pattern = "Gray",
    [string]$FfmpegPath = "ffmpeg",
    [string]$FfprobePath = "ffprobe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = Split-Path -Parent $scriptDir
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repositoryRoot "test_data"
}

$temporaryPath = Join-Path $repositoryRoot ".test-filter-disabled-fixture"
$sourceAvif = Join-Path $temporaryPath "$FixtureId.avif"
$sourceFilter = switch ($Pattern) {
    "Gray" { "color=c=gray:size=16x16:rate=1" }
    "Black" { "color=c=black:size=16x16:rate=1" }
    "VerticalSplit" {
        "color=c=black:size=32x64:rate=1[left];color=c=white:size=32x64:rate=1[right];[left][right]hstack,format=gbrp"
    }
    "TestPattern" { "testsrc=size=64x64:rate=1" }
    "Palette" {
        "nullsrc=size=64x64:rate=1,format=rgb24,geq=r='255*mod(floor(X/4),2)':g='255*mod(floor(Y/4),2)':b='255*mod(floor(X/4)+floor(Y/4),2)',format=gbrp"
    }
}
$enablePalette = [int]($Pattern -eq "Palette")
New-Item -ItemType Directory -Force -Path $temporaryPath | Out-Null

try {
    & $FfmpegPath `
        -v error `
        -f lavfi `
        -i $sourceFilter `
        -frames:v 1 `
        -vf format=gbrp `
        -c:v libaom-av1 `
        -still-picture 1 `
        -usage allintra `
        -cpu-used 8 `
        -crf 0 `
        -pix_fmt gbrp `
        -enable-cdef 0 `
        -enable-restoration 0 `
        -enable-palette $enablePalette `
        -enable-intrabc 0 `
        -enable-filter-intra 0 `
        -enable-angle-delta 0 `
        -enable-cfl-intra 0 `
        -enable-rect-partitions 0 `
        -enable-1to4-partitions 0 `
        -enable-ab-partitions 0 `
        -enable-flip-idtx 0 `
        -use-intra-default-tx-only 1 `
        -color_range pc `
        -colorspace rgb `
        -y `
        $sourceAvif
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg failed while generating the filter-disabled AVIF fixture."
    }

    & (Join-Path $scriptDir "generate_oracles.ps1") `
        -SourceAvif $sourceAvif `
        -OutputDir $OutputDir `
        -FixtureId $FixtureId `
        -FfmpegPath $FfmpegPath `
        -FfprobePath $FfprobePath
}
finally {
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Recurse -Force
    }
}
