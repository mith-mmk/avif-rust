[CmdletBinding()]
param(
    [string]$OracleDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OracleDir)) {
    $OracleDir = Join-Path (Split-Path -Parent $scriptDir) "test_data"
}

$oracleRoot = [System.IO.Path]::GetFullPath($OracleDir)
$manifestPath = Join-Path $oracleRoot "oracles.sources.csv"
$imagesRoot = [System.IO.Path]::GetFullPath((Join-Path $oracleRoot "images"))
$header = "id,source,sha256,plane_format,generated_by"

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "source manifest is missing: $manifestPath"
}

$lines = @(Get-Content -LiteralPath $manifestPath | Where-Object {
    $_ -and $_.Trim() -and -not $_.Trim().StartsWith('#')
})
if ($lines.Count -eq 0 -or $lines[0] -ne $header) {
    throw "unexpected source manifest header in $manifestPath"
}

$ids = [System.Collections.Generic.HashSet[string]]::new()
for ($index = 1; $index -lt $lines.Count; $index++) {
    $lineNumber = $index + 1
    $columns = $lines[$index].Split(',')
    if ($columns.Count -ne 5) {
        throw "source manifest line $lineNumber has $($columns.Count) columns, expected 5"
    }

    $id = $columns[0].Trim()
    $source = $columns[1].Trim()
    $expectedHash = $columns[2].Trim().ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($id) -or -not $ids.Add($id)) {
        throw "source manifest line $lineNumber has a duplicate or empty id"
    }
    if ([string]::IsNullOrWhiteSpace($source)) {
        throw "source manifest line $lineNumber has an empty source"
    }
    if ($expectedHash -notmatch '^[0-9a-f]{64}$') {
        throw "source manifest line $lineNumber has an invalid SHA-256 hash"
    }

    $sourcePath = [System.IO.Path]::GetFullPath((Join-Path $imagesRoot $source))
    if (-not $sourcePath.StartsWith($imagesRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "source manifest line $lineNumber escapes the images directory"
    }
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "source file is missing for $id`: $sourcePath"
    }

    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "source hash mismatch for $id`: expected $expectedHash, actual $actualHash"
    }
}

Write-Host "verified $($ids.Count) AVIF source hashes from $manifestPath"
