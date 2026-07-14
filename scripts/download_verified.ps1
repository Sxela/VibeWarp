[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^https://')]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$Destination,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Fa-f0-9]{64}$')]
    [string]$Sha256
)

$ErrorActionPreference = 'Stop'
$expected = $Sha256.ToUpperInvariant()
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$temporaryPath = "$destinationPath.download"

function Test-ExpectedHash([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }

    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    return $actual -eq $expected
}

if (Test-ExpectedHash $destinationPath) {
    Write-Host "Verified cached download: $destinationPath"
    exit 0
}

if (Test-Path -LiteralPath $destinationPath) {
    Write-Warning "Cached file failed SHA-256 verification; downloading a verified replacement."
}

$parent = Split-Path -Parent $destinationPath
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

try {
    Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $temporaryPath -UseBasicParsing

    if (-not (Test-ExpectedHash $temporaryPath)) {
        $actual = (Get-FileHash -LiteralPath $temporaryPath -Algorithm SHA256).Hash
        throw "SHA-256 mismatch for $Url`nExpected: $expected`nActual:   $actual"
    }

    Move-Item -LiteralPath $temporaryPath -Destination $destinationPath -Force
    Write-Host "Verified SHA-256: $expected"
}
finally {
    Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
}
