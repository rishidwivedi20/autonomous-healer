# Stops the dashboard safely: any fault still in place is undone first, then the dashboard is stopped.
#   .\stop.cmd             stop the dashboard (the sandbox app keeps running)
#   .\stop.ps1 -Sandbox    also stop the sandbox containers (their data is kept; .\deploy\tasks.ps1 down wipes it)
param([switch]$Sandbox)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path $py)) { Write-Host "the Python environment is missing; nothing to stop" -ForegroundColor Yellow; exit 0 }

Write-Host "[1/3] Undoing any fault or fix still in place (kill switch)" -ForegroundColor Cyan
& $py -m chaosai abort

Write-Host "[2/3] Stopping the dashboard" -ForegroundColor Cyan
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "chaosai\s+dashboard" }
if (-not $procs) { Write-Host "no dashboard process found" }
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host "stopped process $($p.ProcessId)" }
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $root "results\dashboard.pid")

Write-Host "[3/3] Clearing the kill switch so the next start is clean" -ForegroundColor Cyan
& $py -m chaosai reset

if ($Sandbox) {
    Write-Host "Stopping the sandbox containers" -ForegroundColor Cyan
    Push-Location (Join-Path $root "deploy")
    docker compose stop
    Pop-Location
}
Write-Host "Done." -ForegroundColor Green
