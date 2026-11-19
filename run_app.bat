@echo off
REM ==========================================================================
REM  Start the full app on Windows (run setup_windows.bat once first):
REM    analysis backend  -> http://localhost:8001   (new window)
REM    frontend (Next.js)-> http://localhost:3333   (new window)
REM    Ollama (native Windows app, port 11434) is used if running; otherwise
REM    explanations fall back to rule-based text.
REM ==========================================================================
setlocal
cd /d "%~dp0"
if not exist ".venv-win\Scripts\python.exe" (
  echo [ERROR] .venv-win not found. Run setup_windows.bat first.
  exit /b 1
)

REM Native Windows Ollama listens on 11434 (the Linux setup used a Docker container on 11435).
REM Use 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first and each
REM request stalls ~2 s before falling back, which made the UI flicker "offline".
if not defined OLLAMA_URL set "OLLAMA_URL=http://127.0.0.1:11434"
REM gemma2:2b fits a 6 GB laptop GPU next to the detectors; use gemma2:9b on bigger GPUs.
if not defined OLLAMA_MODEL set "OLLAMA_MODEL=gemma2:2b"
if not defined FORENSICS_URL set "FORENSICS_URL=http://127.0.0.1:8001"
set "TF_ENABLE_ONEDNN_OPTS=0"
set "PYTHONIOENCODING=utf-8"

where ollama >nul 2>nul && (start "" /b ollama serve >nul 2>nul)

start "Deepfake backend :8001" cmd /k ".venv-win\Scripts\python.exe -m uvicorn backend.forensics_api:app --host 127.0.0.1 --port 8001"
start "Deepfake frontend :3333" cmd /k "cd frontend && npm run dev"

echo Waiting for the models to load (first start can take a minute) ...
for /l %%i in (1,1,90) do (
  curl -s -m 2 http://127.0.0.1:8001/health | findstr /c:"\"engine_loaded\":true" >nul && goto ready
  timeout /t 2 /nobreak >nul
)
echo [WARN] Backend not ready after 3 minutes - check the backend window for errors.
goto open
:ready
echo Backend ready.
:open
start "" http://127.0.0.1:3333
echo App opened at http://localhost:3333  (close the two windows to stop it)
endlocal
