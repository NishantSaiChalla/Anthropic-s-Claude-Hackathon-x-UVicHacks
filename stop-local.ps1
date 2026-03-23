$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Stop-FromPidFile {
    param([string]$PidFile)

    if (-not (Test-Path $PidFile)) {
        return
    }

    $procId = Get-Content $PidFile -ErrorAction SilentlyContinue
    if (-not $procId) {
        return
    }

    try {
        Stop-Process -Id ([int]$procId) -Force -ErrorAction Stop
        Write-Host "[local-run] Stopped PID $procId"
    } catch {
        Write-Host "[local-run] PID $procId was not running"
    }
}

function Stop-ByPort {
    param([int]$Port)

    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($procId in $listeners) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "[local-run] Stopped PID $procId on port $Port"
        } catch {
            Write-Host "[local-run] Could not stop PID $procId on port $Port"
        }
    }
}

$runDir = Join-Path $PSScriptRoot ".run"
Stop-FromPidFile (Join-Path $runDir "backend.pid")
Stop-FromPidFile (Join-Path $runDir "frontend.pid")
Stop-ByPort 8001
Stop-ByPort 3001
Stop-ByPort 3002
