<#
Start the ImagineGeneratorProxy server (PowerShell)

Usage:
  .\start_server.ps1 -Port 8088
  # or rely on the PORT environment variable / config.yaml token
#>

param(
    [int]$Port = 0
)

if ($Port -gt 0) {
    $env:PORT = $Port.ToString()
}

Write-Host "Starting server on port $($env:PORT ?? '8080')..."
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path $venvPython) {
    & $venvPython -m imagine_generator_proxy
} else {
    python -m imagine_generator_proxy
}
