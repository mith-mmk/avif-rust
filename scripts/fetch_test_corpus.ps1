[CmdletBinding()]
param(
    [string]$ManifestPath = "",
    [string]$OutputDir = "",
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path (Get-Location) $Path
}

function Test-CorpusId {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id
    )

    if ($Id -notmatch '^[A-Za-z0-9._-]+$') {
        throw "Corpus id '$Id' is invalid; use only letters, digits, '.', '_' and '-'."
    }
}

$scriptDir = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    $PSScriptRoot
}

if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $scriptDir "test_corpus.csv"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Split-Path -Parent $scriptDir) "test_data"
}

$manifest = Resolve-RelativePath $ManifestPath
$output = Resolve-RelativePath $OutputDir

if (!(Test-Path -LiteralPath $manifest)) {
    throw "Corpus manifest not found: $manifest"
}

$entries = @(Import-Csv -LiteralPath $manifest)
if ($entries.Count -eq 0) {
    Write-Host "No corpus entries in $manifest"
    exit 0
}

New-Item -ItemType Directory -Force -Path $output | Out-Null

foreach ($entry in $entries) {
    foreach ($field in @("id", "url", "sha256", "license", "source")) {
        if ([string]::IsNullOrWhiteSpace($entry.$field)) {
            throw "Corpus entry is missing '$field'."
        }
    }

    Test-CorpusId $entry.id
    $expectedHash = $entry.sha256.ToLowerInvariant()
    if ($expectedHash -notmatch '^[0-9a-f]{64}$') {
        throw "Corpus entry '$($entry.id)' has an invalid SHA-256."
    }

    $target = Join-Path $output $entry.id
    if (!$VerifyOnly -and !(Test-Path -LiteralPath $target)) {
        $partial = "$target.tmp"
        if (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial
        }
        Invoke-WebRequest -Uri $entry.url -OutFile $partial
        Move-Item -LiteralPath $partial -Destination $target
    }

    if (!(Test-Path -LiteralPath $target)) {
        throw "Corpus file is missing: $target"
    }

    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "SHA-256 mismatch for '$($entry.id)': expected $expectedHash, got $actualHash"
    }

    Write-Host "verified $($entry.id) $($entry.license) $($entry.source)"
}
