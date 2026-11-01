<#
  The repo lives on a Google Drive volume (G:), where `npm install` cannot create node_modules
  (no junctions, flaky writes, and thousands of files would be synced). This script mirrors ui/
  to a local NTFS folder and runs an npm script there.

  Usage (from ui/):
    powershell -ExecutionPolicy Bypass -File scripts\sync-local.ps1            # sync + install + dev server
    powershell -ExecutionPolicy Bypass -File scripts\sync-local.ps1 -Task build
    powershell -ExecutionPolicy Bypass -File scripts\sync-local.ps1 -Task none  # only sync
#>
param(
  [string]$Dest = (Join-Path $env:LOCALAPPDATA "tsf-ui"),
  [ValidateSet("dev", "build", "preview", "none")] [string]$Task = "dev"
)
$ErrorActionPreference = "Stop"
$src = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
New-Item -ItemType Directory -Force $Dest | Out-Null

# /MIR keeps the destination identical to the source, except for the excluded directories.
robocopy $src $Dest /MIR /NFL /NDL /NJH /NJS /NP /XD node_modules dist .vite | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }

Push-Location $Dest
try {
  if (-not (Test-Path (Join-Path $Dest "node_modules"))) { npm install }
  switch ($Task) {
    "dev"     { npm run dev }
    "build"   { npm run build }
    "preview" { npm run preview }
    default   { Write-Host "Synced to $Dest" }
  }
} finally { Pop-Location }
