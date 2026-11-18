@echo off
rem Remaining network fetches for the light full-universe backfill, back to back, resumable.
rem Run from anywhere:  scripts\run_network_steps.bat   (or double-click)
rem Step 1: sibling price histories for the within-event test (coarse 3h/12h grains), ~1.5 h at 5 req/s.
rem Step 2: first_seen for candidate wallets only (data\private\first_seen_candidates.parquet), ~0.4 h.
rem Total about 2 hours. Safe to stop (Ctrl+C) and re-run: finished parts are skipped.
rem Progress: logs\event_fetch_full.log and logs\first_seen_candidates.log
cd /d "%~dp0.."
if not exist logs mkdir logs
echo [%date% %time%] step 1/2: event siblings >> logs\network_steps.log
.venv\Scripts\python.exe scripts\event_fetch_full.py >> logs\event_fetch_full.log 2>&1
if errorlevel 1 ( echo [%date% %time%] step 1 FAILED, see logs\event_fetch_full.log >> logs\network_steps.log & goto :end )
echo [%date% %time%] step 2/2: first_seen candidates >> logs\network_steps.log
.venv\Scripts\python.exe scripts\first_seen_candidates.py fetch --threads 3 >> logs\first_seen_candidates.log 2>&1
if errorlevel 1 ( echo [%date% %time%] step 2 FAILED, see logs\first_seen_candidates.log >> logs\network_steps.log & goto :end )
echo [%date% %time%] both steps done >> logs\network_steps.log
:end
echo See logs\network_steps.log
pause
