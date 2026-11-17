<#
.SYNOPSIS
    Smoke-test a running local AIoT web service: model identity, threshold, scoring and SIMULATED labelling.

.DESCRIPTION
    Verifies, against a live server, that
      /health   answers ok with the frozen model loaded and the expected feature count,
      /model    names xgboost_final and lists the 34 features the scorer uses,
      /simulate returns SIMULATED geothermal points with per-point scores and a SYNTHETIC warning,
      /score    produces a real probability (not a stub) for a real building history,
      /         serves the dashboard with the SIMULATED label visible in the HTML.
    Exit code 0 only when every check passes. Run it once, restart the server, run it again.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/smoke_web.ps1
    powershell -ExecutionPolicy Bypass -File scripts/smoke_web.ps1 -Base http://127.0.0.1:8000
#>
[CmdletBinding()]
param(
    [string]$Base = 'http://127.0.0.1:8000',
    [int]$MinFeatures = 30
)

$ErrorActionPreference = 'Stop'
$script:Failed = 0
$script:Passed = 0

function Check {
    param([string]$Name, [scriptblock]$Test)
    try {
        $detail = & $Test
        $script:Passed++
        Write-Host ("  [PASS] {0}{1}" -f $Name, $(if ($detail) { " -> $detail" } else { '' })) -ForegroundColor Green
    } catch {
        $script:Failed++
        Write-Host ("  [FAIL] {0}: {1}" -f $Name, $_.Exception.Message) -ForegroundColor Red
    }
}

function Get-Json { param([string]$Path) Invoke-RestMethod -Uri "$Base$Path" -TimeoutSec 60 }

Write-Host "smoke test against $Base" -ForegroundColor Cyan

# ------------------------------------------------------------------ /health
$health = $null
Check '/health answers ok and loaded the frozen model' {
    $script:health = Get-Json '/health'
    if ($script:health.status -ne 'ok') { throw "status=$($script:health.status)" }
    if ($script:health.model -ne 'xgboost_final') { throw "model=$($script:health.model)" }
    if (-not $script:health.threshold -or $script:health.threshold -le 0) { throw 'no threshold' }
    if ([int]$script:health.n_features -lt $MinFeatures) { throw "n_features=$($script:health.n_features)" }
    "model=$($script:health.model) threshold=$([math]::Round($script:health.threshold,6)) n_features=$($script:health.n_features)"
}

# ------------------------------------------------------------------ /model
$card = $null
Check '/model reports the frozen model, its selection rule and features' {
    $script:card = Get-Json '/model'
    if ($script:card.model -ne 'xgboost_final') { throw "model=$($script:card.model)" }
    if ($script:card.features.Count -lt $MinFeatures) { throw "features=$($script:card.features.Count)" }
    if ([math]::Abs([double]$script:card.threshold - [double]$script:health.threshold) -gt 1e-12) {
        throw 'threshold differs between /health and /model'
    }
    "selected_from=$($script:card.selected_from) features=$($script:card.features.Count) val_roc_auc=$([math]::Round($script:card.validation_metrics.roc_auc,4))"
}

Check '/model carries the SIMULATED / synthetic-data caveat' {
    $cav = ($script:card.caveats -join ' ') + ' ' + $script:card.training_data.dataset
    if ($cav -notmatch 'SIMULATED|synthetic|commercial-building') {
        throw 'no synthetic-data caveat in the model card'
    }
    'caveat present'
}

# ------------------------------------------------------------------ /simulate
$sim = $null
Check '/simulate returns SIMULATED points scored by the same model' {
    $script:sim = Get-Json '/simulate?equipment=pump&limit=300'
    if (-not $script:sim.simulated) { throw 'simulated flag is not true' }
    if ($script:sim.warning -notmatch 'SYNTHETIC') { throw "warning=$($script:sim.warning)" }
    if ($script:sim.points.Count -lt 100) { throw "points=$($script:sim.points.Count)" }
    $scored = @($script:sim.points | Where-Object { $null -ne $_.score })
    if ($scored.Count -ne $script:sim.points.Count) { throw 'some points carry no score' }
    $distinct = ($scored | Select-Object -ExpandProperty score -Unique).Count
    if ($distinct -lt 5) { throw "scores look constant ($distinct distinct values)" }
    if ([math]::Abs([double]$script:sim.threshold - [double]$script:health.threshold) -gt 1e-12) {
        throw 'simulation used a different threshold'
    }
    "equipment=$($script:sim.equipment) points=$($script:sim.points.Count) distinct_scores=$distinct flagged_hours=$(@($scored | Where-Object { $_.score -ge $script:health.threshold }).Count)"
}

Check '/simulate reports fault hours and detection counts (synthetic ground truth)' {
    if ($null -eq $script:sim.detection) { throw 'no detection block' }
    "fault_hours=$($script:sim.detection.fault_hours) flagged=$($script:sim.detection.fault_hours_flagged)"
}

# ------------------------------------------------------------------ /score
Check '/score produces a real probability for a real building history' {
    $listing = Get-Json '/buildings'
    $bid = [int]$listing.buildings[0].building_id
    $series = Get-Json "/buildings/$bid/series?limit=9000"
    $pts = @($series.points | Where-Object { $null -ne $_.v })
    if ($pts.Count -lt 168) { throw "only $($pts.Count) readings available" }
    # /buildings/{id}/series may stride; scoring needs consecutive hours, so take the last
    # 168 readings of the un-strided run when the stride is 1, otherwise say so.
    $tail = @($pts | Select-Object -Last 168)
    $gaps = 0
    for ($i = 1; $i -lt $tail.Count; $i++) {
        $dt = ([datetime]$tail[$i].t) - ([datetime]$tail[$i - 1].t)
        if ([math]::Abs($dt.TotalHours - 1) -gt 0.001) { $gaps++ }
    }
    $history = @($tail | ForEach-Object { @{ timestamp = $_.t; meter_reading = $_.v } })
    $body = @{ building_id = $bid; history = $history } | ConvertTo-Json -Depth 6 -Compress
    $res = Invoke-RestMethod -Uri "$Base/score" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 120
    if ($null -eq $res.score) { throw 'no score in response' }
    if ($res.model -ne 'xgboost_final') { throw "model=$($res.model)" }
    if ($res.alarm -notin @('ANOMALY', 'NORMAL', 'OUTAGE')) { throw "alarm=$($res.alarm)" }
    if ($res.alarm -ne 'OUTAGE') {
        $s = [double]$res.score
        if ($s -le 0 -or $s -ge 1) { throw "score out of range: $s" }
    }
    "building_id=$($res.building_id) timestamp=$($res.timestamp) score=$($res.score) threshold=$($res.threshold) alarm=$($res.alarm) n_history=$($history.Count) non_hourly_gaps=$gaps"
}

# ------------------------------------------------------------------ dashboard
Check 'dashboard HTML labels the data SIMULATED' {
    $html = (Invoke-WebRequest -Uri "$Base/" -TimeoutSec 30 -UseBasicParsing).Content
    if ($html -notmatch 'SIMULATED') { throw 'SIMULATED label missing from dashboard HTML' }
    if ($html -notmatch '/simulate') { throw 'dashboard does not call /simulate' }
    "html=$($html.Length) chars, contains SIMULATED and /simulate"
}

Write-Host ''
if ($script:Failed -eq 0) {
    Write-Host "ALL CHECKS PASSED ($($script:Passed)/$($script:Passed))" -ForegroundColor Green
    exit 0
}
Write-Host "FAILURES: $($script:Failed) of $($script:Passed + $script:Failed) checks" -ForegroundColor Red
exit 1
