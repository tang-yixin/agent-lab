# -*- powershell -*-
# Start the FastAPI backend (frontend + Agent).
# Auto-locates the project root and the Python interpreter,
# so it works on any machine as long as the folder layout is unchanged.

$ErrorActionPreference = "Stop"

# Project root = parent of this script's folder (scripts/)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "src\backend"

# Prefer the project virtualenv; fall back to system `python`
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
    Write-Host "Using venv Python: $Python"
} else {
    $Python = "python"
    Write-Host "Using system Python (venv not found)"
}

Write-Host "Backend dir: $BackendDir"
Write-Host "Starting backend at http://127.0.0.1:8000 ..."

& $Python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir $BackendDir
