$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-PortInUse {
    param([int]$Port)

    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $listener
}

function Wait-ForHttp {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds 750
        }
    }

    return $false
}

function Write-Status {
    param([string]$Message)
    Write-Host "[local-run] $Message"
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python virtualenv not found at .venv\Scripts\python.exe"
}

$node = (Get-Command node -ErrorAction Stop).Source
if (-not (Test-Path (Join-Path $PSScriptRoot "node_modules"))) {
    throw "node_modules is missing. Run npm install first."
}

$runDir = Join-Path $PSScriptRoot ".run"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$backendLog = Join-Path $runDir "backend.log"
$backendPidFile = Join-Path $runDir "backend.pid"
$frontendPidFile = Join-Path $runDir "frontend.pid"

foreach ($port in 8001, 3000, 3001, 3002) {
    $existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Status "Port $port is already in use by PID(s): $($existing.OwningProcess -join ', ')"
    }
}

$frontendPort = if (-not (Test-PortInUse 3000)) { 3000 } elseif (-not (Test-PortInUse 3002)) { 3002 } else { 3001 }

function Stop-ListeningPort {
    param([int]$Port)

    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($procId in $listeners) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Status "Stopped existing PID $procId on port $Port"
        } catch {
            Write-Status "Could not stop existing PID $procId on port $Port"
        }
    }
}

Stop-ListeningPort 8001
Stop-ListeningPort $frontendPort

Write-Status "Starting backend on http://127.0.0.1:8001"
$backendCommand = @"
Set-Location '$PSScriptRoot'
& '$python' -m uvicorn main:app --host 127.0.0.1 --port 8001 --app-dir backend
"@
$backendProc = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand `
    -PassThru
$backendProc.Id | Set-Content $backendPidFile

Write-Status "Starting frontend on http://localhost:$frontendPort"
$frontendCommand = @"
Set-Location '$PSScriptRoot'
`$env:PORT = '$frontendPort'
& '$node' 'server.js'
"@
$frontendProc = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $frontendCommand `
    -PassThru
$frontendProc.Id | Set-Content $frontendPidFile

$backendReady = Wait-ForHttp -Url "http://127.0.0.1:8001/docs" -TimeoutSeconds 240
$frontendReady = Wait-ForHttp -Url "http://127.0.0.1:$frontendPort/api/health" -TimeoutSeconds 45

if (-not $backendReady) {
    Write-Status "Backend did not become ready in time."
}

if (-not $frontendReady) {
    Write-Status "Frontend did not become ready in time."
}

Write-Host ""
Write-Host "Frontend URL: http://localhost:$frontendPort"
Write-Host "Backend WS: ws://localhost:8001/ws"
Write-Host "Backend window PID: $($backendProc.Id)"
Write-Host "Frontend window PID: $($frontendProc.Id)"
