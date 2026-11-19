[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 5187,
    [string]$DataDir = '.local',
    [string]$PythonPath = (Join-Path $env:USERPROFILE 'miniconda3\envs\YoloWebAgentStarter\python.exe')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python was not found at $PythonPath. Supply an existing Python executable with -PythonPath."
}

$python = (Resolve-Path -LiteralPath $PythonPath).Path
Push-Location -LiteralPath $PSScriptRoot
try {
    & $python -B (Join-Path $PSScriptRoot 'server.py') --port $Port --data-dir $DataDir
    if ($LASTEXITCODE -ne 0) {
        throw "The workbench exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
