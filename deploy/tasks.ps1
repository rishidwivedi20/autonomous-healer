# Windows equivalent of the Makefile (no `make` on Windows). Usage: .\deploy\tasks.ps1 up|down|status
param([Parameter(Mandatory)][ValidateSet("up", "down", "status")][string]$Task)
Push-Location $PSScriptRoot          # return to the caller's folder afterwards, so `.\chaos.cmd` still works
try {
    switch ($Task) {
        "up"     { docker compose up -d --wait; Write-Host "vote: http://localhost:8080  result: http://localhost:8081" }
        "down"   { docker compose down -v }
        "status" { docker compose ps }
    }
} finally {
    Pop-Location
}
