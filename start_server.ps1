<#
Start the ImagineGeneratorProxy server (PowerShell)

Usage:
  .\start_server.ps1 -Token '<token>' -Port 8080
  # or rely on env vars:
  $env:NOVELAI_ACCESS_TOKEN='token'; .\start_server.ps1
#>

param(
    [string]$Token = $env:NOVELAI_ACCESS_TOKEN,
    [string]$Email = $env:NOVELAI_EMAIL,
    [string]$Password = $env:NOVELAI_PASSWORD,
    [int]$Port = 8080
)

if ($Token) {
    $env:NOVELAI_ACCESS_TOKEN = $Token
} elseif ($Email -and $Password) {
    $env:NOVELAI_EMAIL = $Email
    $env:NOVELAI_PASSWORD = $Password
}

$env:PORT = $Port.ToString()

Write-Host "Starting server on port $env:PORT..."
$venv_python = Join-Path $PSScriptRoot 'imagine_generator_proxy\Scripts\python.exe'
if (Test-Path $venv_python) {
    & $venv_python -m imagine_generator_proxy.main
} else {
    python -m imagine_generator_proxy.main
}
