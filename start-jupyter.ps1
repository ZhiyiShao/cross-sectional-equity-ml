$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:JUPYTER_RUNTIME_DIR = Join-Path $PSScriptRoot '.jupyter-runtime'
& (Join-Path $PSScriptRoot '.venv/Scripts/python.exe') -m jupyterlab @args
exit $LASTEXITCODE
