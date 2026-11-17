@echo off
rem Unattended pilot: start the model server, search for a set number of hours, write the report, stop the server.
rem   overnight.cmd RUN HOURS [STYLE]    e.g.  overnight.cmd pilot7 6 both
rem STYLE is long (default), day or both: how long positions are held (generate.py --style).
rem Docker Desktop must already be running. Output goes to runs\RUN.log and runs\RUN_report.md.
rem It does not depend on the window or session that started it, so it can run while you sleep.
cd /d "%~dp0"
set RUN=%1
set HOURS=%2
set STYLE=%3
if "%RUN%"=="" set RUN=pilot
if "%HOURS%"=="" set HOURS=5
if "%STYLE%"=="" set STYLE=long
set PYTHONUTF8=1
rem /b: no window of its own. pilot9 lost its server after 1.5 hours, most likely by its minimized window being closed.
start "llama-server" /b wsl -d Ubuntu-24.04 -- bash -lc "exec ~/llama.cpp/build/bin/llama-server -m ~/models/gemma-4-12B-it-Q4_K_M.gguf -ngl 99 -c 16384 -np 1 --reasoning-budget 0 --host 127.0.0.1 --port 8080"
set /a TRIES=0
:wait
curl -sf -m 3 http://127.0.0.1:8080/health >nul 2>&1 && goto ready
set /a TRIES+=1
if %TRIES% gtr 120 (echo the model server did not come up within 10 minutes >> runs\%RUN%.log & goto stop)
ping -n 6 127.0.0.1 >nul
goto wait
:ready
.venv\Scripts\python -u generate.py --run %RUN% --hours %HOURS% --style %STYLE% --review-every 20 >> runs\%RUN%.log 2>&1
echo exit code %errorlevel% >> runs\%RUN%.log
:stop
wsl -d Ubuntu-24.04 -- pkill llama-server
