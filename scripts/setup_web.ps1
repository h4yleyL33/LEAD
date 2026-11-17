<#
.SYNOPSIS
    One-time setup for the local AIoT web service: environment, data, model, simulation artefacts.

.DESCRIPTION
    Run this once after cloning (or after deleting artefacts). It
    1. creates .venv if it is missing and installs requirements.txt,
    2. builds every artefact the web service needs, by calling
       `python scripts/prepare_artifacts.py` (which runs the same commands as scripts/run_all.py,
       restricted to the stages the three deliverables depend on),
    3. verifies that model, feature order, threshold and metrics describe one experiment.

    Training happens here — never at server start. To only fetch/build data and the frozen model
    without the full evaluation set, pass -Deliverables model.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/setup_web.ps1

.EXAMPLE
    # Rerun the build from scratch (e.g. after a failed download)
    powershell -ExecutionPolicy Bypass -File scripts/setup_web.ps1 -Force

.EXAMPLE
    # Read-only: confirm the model, feature order, threshold and metrics belong to one experiment
    powershell -ExecutionPolicy Bypass -File scripts/setup_web.ps1 -Check
#>
[CmdletBinding()]
param(
    [switch]$Force,                       # rebuild artefacts even when the files exist
    [ValidateSet('all', 'model', 'web')]
    [string]$Deliverables = 'all',        # all = model + evaluation + simulation; model = minimum; web = wire up ui only
    [switch]$SkipInstall,                 # do not touch .venv / pip (offline runs)
    [switch]$Check                        # read-only: verify the artefacts and exit (writes nothing)
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot          # repository root
Set-Location $Root
Write-Host "repository: $Root"

$VenvPy = Join-Path $Root '.venv\Scripts\python.exe'
$SysPy = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $SysPy) { throw "python was not found on PATH; install Python 3.11+ and retry" }

function Invoke-Step {
    param([string]$Label, [scriptblock]$Body)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE)" }
}

# ---------------------------------------------------------------- 0. read-only check
if ($Check) {
    if (-not (Test-Path $VenvPy)) {
        throw ".venv\Scripts\python.exe is missing; run scripts\setup_web.ps1 once (without -Check)"
    }
    Write-Host "`n=== verify artefacts (read-only) ===" -ForegroundColor Cyan
    $Latex = $null
    foreach ($Engine in 'pdflatex', 'latexmk') {
        $Cmd = Get-Command $Engine -ErrorAction SilentlyContinue
        if ($Cmd) { $Latex = $Engine; break }
    }
    if ($Latex) {
        Write-Host "latex engine: $Latex ($($Cmd.Source))" -ForegroundColor Gray
    } else {
        Write-Host "latex engine: none on PATH - paper/main.tex stays source-only (see BLOCKERS.md B-003 for the TinyTeX install)" -ForegroundColor Yellow
    }
    & $VenvPy 'scripts/prepare_artifacts.py' --check
    $Code = $LASTEXITCODE
    if ($null -eq $Code) { $Code = 0 }
    if ($Code -ne 0) { throw "artefact check failed (exit $Code); run scripts\setup_web.ps1" }
    Write-Host "`nartefacts verified; no artefact was rebuilt (the check log is results\metrics\artifact_check.json)." -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------- 1. environment
if (-not $SkipInstall) {
    if (-not (Test-Path $VenvPy)) {
        Invoke-Step 'create .venv' { & $SysPy -m venv (Join-Path $Root '.venv') }
    }
    Invoke-Step 'install requirements' {
        & $VenvPy -m pip install --upgrade pip --quiet
        & $VenvPy -m pip install -r (Join-Path $Root 'requirements.txt') --quiet
    }
}
if (-not (Test-Path $VenvPy)) { throw ".venv\Scripts\python.exe is missing; rerun without -SkipInstall" }
$Py = $VenvPy

# ---------------------------------------------------------------- 2. artefacts
$ForceFlag = @()
if ($Force) { $ForceFlag = @('--force') }

switch ($Deliverables) {
    'model' {
        # dataset -> features -> splits -> tune -> train default/tuned/weighted -> freeze xgb_final.json
        Invoke-Step 'build data + frozen model' {
            & $Py 'scripts/prepare_artifacts.py' --only 'data,features,splits,tune,train_primary,train_tuned,train_weighted,select' @ForceFlag
        }
    }
    'web' {
        Invoke-Step 'check existing artefacts' { & $Py 'scripts/prepare_artifacts.py' --check }
    }
    default {
        Invoke-Step 'build all artefacts (data, model, evaluation, simulation, paper)' {
            & $Py 'scripts/prepare_artifacts.py' @ForceFlag
        }
    }
}

# ---------------------------------------------------------------- 3. verify
Invoke-Step 'verify artefacts' { & $Py 'scripts/prepare_artifacts.py' --check }

Write-Host @"

setup complete. next:
    powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1
    then open http://127.0.0.1:8000/  (dashboard)  or  http://127.0.0.1:8000/docs
"@ -ForegroundColor Green
