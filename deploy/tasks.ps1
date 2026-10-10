# Windows equivalent of the Makefile (no `make` on Windows). Usage: .\tasks.ps1 up|down|status
param([Parameter(Mandatory)][ValidateSet("up", "down", "status")][string]$Task)
Set-Location $PSScriptRoot
switch ($Task) {
    "up"     { docker compose up -d --wait; Write-Host "vote: http://localhost:8080  result: http://localhost:8081" }
    "down"   { docker compose down -v }
    "status" { docker compose ps --format "table {{.Service}}\t{{.State}}\t{{.Health}}\t{{.Ports}}" }
}
