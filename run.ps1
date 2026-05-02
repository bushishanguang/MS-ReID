param(
    [ValidateSet("all", "backend", "frontend")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendDir = Join-Path $Root "app\frontend"
$ScriptPath = $MyInvocation.MyCommand.Path

function Start-Backend {
    Set-Location $Root

    $HostName = "127.0.0.1"
    $Port = "8000"
    $AppModule = "app.backend.main:app"

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv run uvicorn $AppModule --host $HostName --port $Port --reload
        exit $LASTEXITCODE
    }

    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        & $VenvPython -m uvicorn $AppModule --host $HostName --port $Port --reload
        exit $LASTEXITCODE
    }

    throw "uv or .venv\Scripts\python.exe was not found. Run 'uv sync' in the project root first."
}

function Start-Frontend {
    if (-not (Test-Path $FrontendDir)) {
        throw "Frontend directory was not found: $FrontendDir"
    }
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw "npm was not found. Install Node.js first."
    }

    Set-Location $FrontendDir
    npm run dev -- --host 127.0.0.1
    exit $LASTEXITCODE
}

if ($Target -eq "backend") {
    Start-Backend
}

if ($Target -eq "frontend") {
    Start-Frontend
}

$ShellPath = (Get-Process -Id $PID).Path

Start-Process -FilePath $ShellPath -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $ScriptPath,
    "-Target",
    "backend"
) | Out-Null

Start-Process -FilePath $ShellPath -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $ScriptPath,
    "-Target",
    "frontend"
) | Out-Null

Write-Host "Started backend and frontend windows."
Write-Host "Backend: http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"
