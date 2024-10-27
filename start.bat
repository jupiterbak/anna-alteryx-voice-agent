@echo off
setlocal enabledelayedexpansion

echo Starting LiveKit system...

:: Start LiveKit server
start "LiveKit Server" cmd /c "bin\livekit-server.exe --dev"
set livekit_pid=!ERRORLEVEL!

:: Wait for a moment to ensure the server is up
timeout /t 5 > nul

:: Start the agent (main.py)
start "LiveKit Agent" cmd /c "C:\ProgramData\Miniconda3\envs\voice-agent\python.exe c:\DATA\00_Workspace\Github\anna-alteryx-voice-agent\main.py dev"
set agent_pid=!ERRORLEVEL!

:: Wait for a moment to ensure the agent is up
timeout /t 5 > nul

:: Start the UI
pushd ui\anna-assistant
start "LiveKit UI" cmd /c "pnpm dev"
set ui_pid=!ERRORLEVEL!
popd

echo All components started. Press any key to stop all processes and exit.
pause > nul

:: Kill all started processes
taskkill /F /PID %livekit_pid% 2>nul
taskkill /F /PID %agent_pid% 2>nul
taskkill /F /PID %ui_pid% 2>nul

echo All processes have been terminated.
