@echo off
setlocal
cd /d "%~dp0"
start "" python run_frog_study_subprocess.py %*
