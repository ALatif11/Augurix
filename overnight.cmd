@echo off
rem Unattended pilot: start the model server, search for a set number of hours, write the report, stop the server.
rem   overnight.cmd RUN HOURS            e.g.  overnight.cmd pilot6 6
rem Docker Desktop must already be running. Output goes to runs\RUN.log and runs\RUN_report.md.
rem It does not depend on the window or session that started it, so it can run while you sleep.
cd /d "%~dp0"
set RUN=%1
set HOURS=%2
if "%RUN%"=="" set RUN=pilot
if "%HOURS%"=="" set HOURS=5
set PYTHONUTF8=1
start "llama-server" /min wsl -d Ubuntu-24.04 -- bash -lc "exec ~/llama.cpp/build/bin/llama-server -m ~/models/gemma-4-12B-it-Q4_K_M.gguf -ngl 99 -c 16384 -np 1 --reasoning-budget 0 --host 127.0.0.1 --port 8080"
set /a TRIES=0
:wait
curl -sf -m 3 http://127.0.0.1:8080/health >nul 2>&1 && goto ready
set /a TRIES+=1
if %TRIES% gtr 120 (echo the model server did not come up within 10 minutes >> runs\%RUN%.log & goto stop)
ping -n 6 127.0.0.1 >nul
goto wait
:ready
.venv\Scripts\python -u generate.py --run %RUN% --hours %HOURS% --review-every 20 >> runs\%RUN%.log 2>&1
echo exit code %errorlevel% >> runs\%RUN%.log
:stop
wsl -d Ubuntu-24.04 -- pkill llama-server
