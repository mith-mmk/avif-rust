[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$FfmpegPath = "ffmpeg",
    [string]$FfprobePath = "ffprobe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Executable {
    param([Parameter(Mandatory = $true)][string]$Path)
    $command = Get-Command $Path -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Required executable was not found: $Path"
    }
}

Assert-Executable -Path $FfmpegPath
Assert-Executable -Path $FfprobePath

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$generator = Join-Path $scriptDir "generate_filter_disabled_oracle.ps1"
$fullOracleGenerator = Join-Path $scriptDir "generate_oracles.ps1"
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Split-Path -Parent $scriptDir) "test_data"
}

$fixtures = @(
    @{ Id = "BlackLossless"; Pattern = "Black"; Width = 64; Height = 64 },
    @{ Id = "filter-disabled-gbr"; Pattern = "Gray"; Width = 16; Height = 16 },
    @{ Id = "filter-disabled-residual"; Pattern = "Black"; Width = 16; Height = 16 },
    @{ Id = "filter-disabled-partition"; Pattern = "VerticalSplit"; Width = 64; Height = 64 },
    @{ Id = "filter-disabled-directional"; Pattern = "TestPattern"; Width = 64; Height = 64 },
    @{ Id = "filter-disabled-palette"; Pattern = "Palette"; Width = 64; Height = 64 }
)

foreach ($fixture in $fixtures) {
    & $generator `
        -OutputDir $OutputDir `
        -FixtureId $fixture.Id `
        -Pattern $fixture.Pattern `
        -Width $fixture.Width `
        -Height $fixture.Height `
        -FfmpegPath $FfmpegPath `
        -FfprobePath $FfprobePath
    if ($LASTEXITCODE -ne 0) {
        throw "Fixture generation failed for $($fixture.Id)."
    }
}

$wml2ViewerSource = Join-Path $repositoryRoot "samples/WML2Viewer.avif"
& $fullOracleGenerator `
    -SourceAvif $wml2ViewerSource `
    -OutputDir $OutputDir `
    -FixtureId "WML2Viewer" `
    -FfmpegPath $FfmpegPath `
    -FfprobePath $FfprobePath `
    -RegisterInStrictManifest
if ($LASTEXITCODE -ne 0) {
    throw "Fixture generation failed for WML2Viewer."
}

Write-Host "generated $($fixtures.Count + 1) strict AVIF oracle fixtures in $OutputDir"
