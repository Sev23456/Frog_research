@echo off
setlocal
cd /d "%~dp0"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i

set "OUT=%~dp0benchmark_results\launch_logs\frog_study_%TS%.out.log"
set "ERR=%~dp0benchmark_results\launch_logs\frog_study_%TS%.err.log"

start "" /b "%~dp0run_frog_study_full.cmd" %* 1>"%OUT%" 2>"%ERR%"

echo OUT=%OUT%
echo ERR=%ERR%
