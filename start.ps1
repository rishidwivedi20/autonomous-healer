# One command that brings everything up: the sandbox app, the live dashboard (with the injecting-agent panel), and the browser.
#   .\start.cmd            (or  .\start.ps1)
#   .\start.ps1 -Fresh     rebuild the sandbox from scratch first (wipes its demo votes)
#   .\start.ps1 -NoBrowser do not open the browser
# Stop it again with .\stop.cmd
param([switch]$Fresh, [switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$results = Join-Path $root "results"
$env:PYTHONIOENCODING = "utf-8"

function Step($n, $msg) { Write-Host ""; Write-Host "[$n/5] $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host ""; Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

Step 1 "Checking the tools"
if (-not (Test-Path $py)) { Fail "the Python environment is missing ($py). Create it first: python -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt" }
cmd /c "docker info >nul 2>&1"
if ($LASTEXITCODE -ne 0) { Fail "Docker is not running. Start Docker Desktop, wait until it says it is running, then run this again." }
$port = (& $py -c "import chaosai.config as c; print(c.load().dashboard.port)").Trim()
Write-Host "Docker is running; dashboard port $port"

Step 2 "Starting the sandbox app (vote, redis, worker, db, result)"
if ($Fresh) { & (Join-Path $root "deploy\tasks.ps1") down }
& (Join-Path $root "deploy\tasks.ps1") up
if ($LASTEXITCODE -ne 0) { Fail "the sandbox did not become healthy. See: .\deploy\tasks.ps1 status" }

Step 3 "Making sure no stale safety state is left over"
$stale = (Test-Path (Join-Path $results "KILL")) -or (Test-Path (Join-Path $results "active_faults.json")) -or (Test-Path (Join-Path $results "active_fixes.json"))
$listening = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($stale -and -not $listening) {
    Write-Host "found leftover kill-switch or undo files from an earlier run: restoring everything and clearing them"
    & $py -m chaosai reset
    if ($LASTEXITCODE -ne 0) { Fail "reset reported problems; run  .\chaos.cmd reset  and read its output" }
} elseif ($stale) {
    Write-Host "leftover safety files exist, but a dashboard is already running: press Reset on its page if the kill switch shows as engaged"
} else {
    Write-Host "clean"
}

Step 4 "Starting the dashboard"
$url = "http://127.0.0.1:$port/"
if ($listening) {
    Write-Host "something is already listening on port $port (probably the dashboard from an earlier start): using it"
} else {
    New-Item -ItemType Directory -Force $results | Out-Null
    $log = Join-Path $results "dashboard.log"
    $err = Join-Path $results "dashboard.err.log"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "-m", "chaosai", "dashboard") -WorkingDirectory $root `
        -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
    Set-Content -Path (Join-Path $results "dashboard.pid") -Value $p.Id
    $up = $false
    for ($i = 0; $i -lt 40 -and -not $up; $i++) {
        Start-Sleep -Seconds 1
        if ($p.HasExited) { Fail "the dashboard exited at start-up. Read $err" }
        try { $up = (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch { $up = $false }
    }
    if (-not $up) { Fail "the dashboard did not answer within 40 s. Read $err" }
    Write-Host "dashboard started (process $($p.Id)); log: $log"
}

Step 5 "Checking that everything answers"
$ok = $true
foreach ($u in @(@("dashboard", $url), @("vote app", "http://127.0.0.1:8080/"), @("results page", "http://127.0.0.1:8081/"))) {
    try { $c = (Invoke-WebRequest -Uri $u[1] -UseBasicParsing -TimeoutSec 5).StatusCode } catch { $c = "no answer" }
    if ($c -ne 200) { $ok = $false }
    Write-Host ("  {0,-13} {1}   {2}" -f $u[0], $u[1], $c)
}
if (-not $ok) { Fail "something does not answer (see above)" }

Write-Host ""
Write-Host "Everything is up." -ForegroundColor Green
Write-Host "  Dashboard : $url   (press 'Start the agent' to watch it inject faults by itself)"
Write-Host "  Vote app  : http://127.0.0.1:8080/"
Write-Host "  Results   : http://127.0.0.1:8081/"
Write-Host "  Stop all  : .\stop.cmd        (add -Sandbox to stop the app containers too)"
if (-not $NoBrowser) { Start-Process $url }
