<#
Start the ImagineGeneratorProxy server (PowerShell)

Usage:
  .\start_server.ps1              # port from $env:PORT or 8080
  .\start_server.ps1 -Port 8088   # explicit port

Auth comes from data\config.yaml (token) or the NOVELAI_TOKEN /
NOVELAI_EMAIL + NOVELAI_PASSWORD environment variables.

If Windows blocks the script ("running scripts is disabled"), run:
  powershell -ExecutionPolicy Bypass -File .\start_server.ps1
#>

param(
    [int]$Port = 0
)

if ($Port -gt 0) {
    $env:PORT = $Port.ToString()
}

$displayPort = $(if ($env:PORT) { $env:PORT } else { '8080' })
Write-Host "Starting server on port $displayPort..."

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Warning "No .venv found at $venvPython - falling back to system python."
}

Set-Location $PSScriptRoot
if (Test-Path $venvPython) {
    & $venvPython -m imagine_generator_proxy
} else {
    & python -m imagine_generator_proxy
}
