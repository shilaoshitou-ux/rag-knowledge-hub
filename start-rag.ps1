# Run from any PowerShell directory. Always use http://localhost:8507.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$entrypoint = Join-Path $root 'src\asgi_app.py'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python virtual environment is missing. Install requirements in .venv first.'
}
$listener = Get-NetTCPConnection -State Listen -LocalPort 8507 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if ($process.CommandLine -like "*$entrypoint*") {
        Write-Host 'RAG is already running: http://localhost:8507/'
        Write-Host 'Refresh that tab; do not start another server.'
        return
    }
    throw 'Port 8507 is used by an older app or another program. Stop it before starting the new persistent-login entrypoint.'
}
Set-Location -LiteralPath $root
Write-Host 'Open http://localhost:8507/ (keep this PowerShell window open).'
& $python -m streamlit run $entrypoint --server.address=127.0.0.1 --server.port=8507 --server.headless=true
