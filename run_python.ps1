$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvSitePackages = Join-Path $ProjectRoot ".venv\Lib\site-packages"
$PyvenvConfig = Join-Path $ProjectRoot ".venv\pyvenv.cfg"

if (-not (Test-Path $VenvSitePackages)) {
    throw "Nao encontrei .venv\Lib\site-packages. Crie a venv e instale requirements.txt antes de rodar."
}

$PythonHome = $null
if (Test-Path $PyvenvConfig) {
    $HomeLine = Get-Content $PyvenvConfig | Where-Object { $_ -like "home = *" } | Select-Object -First 1
    if ($HomeLine) {
        $PythonHome = $HomeLine.Split("=", 2)[1].Trim()
    }
}

$PythonExe = if ($PythonHome) { Join-Path $PythonHome "python.exe" } else { Join-Path $ProjectRoot ".venv\Scripts\python.exe" }

if (-not (Test-Path $PythonExe)) {
    throw "Nao encontrei o Python em: $PythonExe"
}

$env:PYTHONPATH = $VenvSitePackages
& $PythonExe @args
