$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment Python not found at $pythonExe"
}

$streamlitArgs = @(
    "-m", "streamlit", "run", "app.py",
    "--server.headless", "true"
)

$mlflowArgs = @(
    "-m", "mlflow", "ui",
    "--backend-store-uri", ".\mlruns"
)

$streamlitProcess = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $streamlitArgs `
    -WorkingDirectory $projectRoot `
    -PassThru

$mlflowProcess = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $mlflowArgs `
    -WorkingDirectory $projectRoot `
    -PassThru

Write-Host "Streamlit started with PID $($streamlitProcess.Id)"
Write-Host "MLflow UI started with PID $($mlflowProcess.Id)"
Write-Host "Open Streamlit: http://localhost:8501"
Write-Host "Open MLflow UI: http://localhost:5000"
