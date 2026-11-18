<#
.SYNOPSIS
    Start the local AIoT web service on http://127.0.0.1:8000 using the frozen model.

.DESCRIPTION
    Loads models/xgb_final.json through src.inference.service.AnomalyScorer at startup — this
    command never trains. It first checks the artefacts the service needs; if any is missing it
    names the file and the command that rebuilds it, then exits without binding a port.

    By default the server is started detached (its output goes to logs\web_server.log), the script
    waits until /health answers, prints the /health and /model payloads, and returns. Use
    -Foreground to keep it attached to this console (Ctrl+C stops it) and -Stop to stop a
    previously started server.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1 -Foreground
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$BindHost = '127.0.0.1',
    [switch]$Foreground,      # run attached to this console instead of detaching
    [switch]$Stop,            # stop a server started by this script
    [switch]$NoBrowser,       # do not open the dashboard after a successful start
    [int]$WaitSeconds = 90    # how long to wait for /health
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$PidFile = Join-Path $Root 'logs\web_server.pid'
$LogFile = Join-Path $Root 'logs\web_server.log'
$Base = "http://${BindHost}:${Port}"

function Get-ListenerProcess {
    if (-not (Test-Path $PidFile)) { return $null }
    $id = (Get-Content $PidFile -First 1).Trim()
    if (-not $id) { return $null }
    $p = Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue
    return $p
}

# ---------------------------------------------------------------- stop
if ($Stop) {
    $p = Get-ListenerProcess
    if ($p) {
        # Uvicorn can be a child of the python parent recorded in the pidfile.
        # Kill the tree (parent + children) so port is actually freed.
        & taskkill /T /F /PID $p.Id 2>&1 | Out-Null
        # Fallback: also kill any python process holding the port.
        $conns = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
                 Where-Object { $_.LocalPort -eq $Port }
        foreach ($c in $conns) {
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        }
        Write-Host "stopped web server (pid $($p.Id))"
    } else {
        Write-Host 'no running web server recorded in logs\web_server.pid'
    }
    Remove-Item $PidFile -ErrorAction SilentlyContinue
    return
}

# Reuse a healthy AIoT service; opening the dashboard must not reset its live buffer.
try {
    $health = Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 2
} catch { $health = $null }
if ($health -and $health.status -eq 'ok' -and $health.model -eq 'xgboost_final') {
    Write-Host "web service already up: $Base/"
    if (-not $NoBrowser) { Start-Process "$Base/" }
    return
}

# ---------------------------------------------------------------- artefacts
$VenvPy = Join-Path $Root '.venv\Scripts\python.exe'
if (Test-Path $VenvPy) { $Py = $VenvPy } else { $Py = (Get-Command python).Source }

Write-Host 'checking artefacts the service needs ...'
$missing = & $Py 'scripts/prepare_artifacts.py' --missing
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Warning 'model or data artefacts are missing; the server was not started.'
    Write-Host 'missing files:'
    foreach ($m in $missing) { Write-Host "  $m" }
    Write-Host @"

recover (builds data, model, evaluation and simulation artefacts):
    powershell -ExecutionPolicy Bypass -File scripts/setup_web.ps1

recover just the frozen model (dataset -> features -> train -> select):
    powershell -ExecutionPolicy Bypass -File scripts/setup_web.ps1 -Deliverables model

details:
    & '$Py' scripts/prepare_artifacts.py --list      # stage -> output files
"@
    exit 1
}

# ---------------------------------------------------------------- start
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'logs') | Out-Null
$StdinFile = Join-Path $Root 'logs\web_server.stdin'
if (-not (Test-Path $StdinFile)) { New-Item -ItemType File -Path $StdinFile | Out-Null }
$uvArgs = @('-m', 'uvicorn', 'backend.app.main:app', '--host', $BindHost, '--port', "$Port")

if ($Foreground) {
    Write-Host "starting attached: python $($uvArgs -join ' ')" -ForegroundColor Cyan
    Write-Host "dashboard: $Base/    docs: $Base/docs"
    & $Py @uvArgs
    return
}

Write-Host "starting: python $($uvArgs -join ' ')  (log: logs\web_server.log)" -ForegroundColor Cyan
$proc = Start-Process -FilePath $Py -ArgumentList $uvArgs -WorkingDirectory $Root `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError (Join-Path $Root 'logs\web_server.err.log') `
    -RedirectStandardInput $StdinFile `
    -WindowStyle Hidden -PassThru

# ---------------------------------------------------------------- wait for /health
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$health = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 700
    if ($proc.HasExited) { break }
    try {
        $health = Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 5
        break
    } catch { }
}

if (-not $health) {
    Write-Host ''
    Write-Warning "server did not answer $Base/health within $WaitSeconds s."
    Write-Host '--- logs\web_server.log (tail) ---'
    if (Test-Path $LogFile) { Get-Content $LogFile -Tail 25 }
    if (Test-Path (Join-Path $Root 'logs\web_server.err.log')) {
        Write-Host '--- logs\web_server.err.log (tail) ---'
        Get-Content (Join-Path $Root 'logs\web_server.err.log') -Tail 25
    }
    exit 1
}

# Record the PID that actually holds the port, not the Start-Process wrapper.
$listener = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalPort -eq $Port } |
            Select-Object -First 1
if ($listener) {
    $listener.OwningProcess | Set-Content $PidFile
} else {
    $proc.Id | Set-Content $PidFile
}

Write-Host ''
Write-Host "web service up (pid $($proc.Id))" -ForegroundColor Green
Write-Host "  dashboard: $Base/"
Write-Host "  docs:      $Base/docs"
Write-Host ''
Write-Host '/health:'
$health | ConvertTo-Json -Depth 6
Write-Host ''
Write-Host '/model:'
(Invoke-RestMethod -Uri "$Base/model" -TimeoutSec 10) | ConvertTo-Json -Depth 6
Write-Host ''
Write-Host "stop with: powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1 -Stop"

if (-not $NoBrowser) { Start-Process "$Base/" }
