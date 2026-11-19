@echo off
REM ==========================================================================
REM  One-time setup on Windows: Python 3.10 venv (.venv-win) + frontend.
REM  Needs: Python 3.10 (python.org or "py -3.10"), Node.js 18+, optional Ollama.
REM ==========================================================================
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && (py -3.10 -c "import sys" >nul 2>nul && set "PY=py -3.10")
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PY (
  echo [ERROR] Python 3.10 not found. Install it from https://www.python.org/downloads/release/python-31011/
  exit /b 1
)
echo Using Python: %PY%

if not exist ".venv-win\Scripts\python.exe" (
  echo Creating virtual environment .venv-win ...
  %PY% -m venv .venv-win || exit /b 1
)
set "VPY=.venv-win\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip

REM ---- PyTorch: CUDA build if an NVIDIA GPU is present, else CPU ----
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo NVIDIA GPU found - installing CUDA build of PyTorch
  "%VPY%" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || exit /b 1
) else (
  echo No NVIDIA GPU - installing CPU build of PyTorch
  "%VPY%" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu || exit /b 1
)

"%VPY%" -m pip install -r requirements-windows.txt || exit /b 1
"%VPY%" -c "import torch, tensorflow, retinaface, deep_sort_realtime, cv2; print('Python env OK - torch', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit /b 1

REM ---- frontend ----
where npm >nul 2>nul || (echo [WARN] npm not found - install Node.js 18+ to run the UI & goto :ollama)
pushd frontend
call npm ci || call npm install
popd

:ollama
REM ---- optional: local LLM for explanations ----
where ollama >nul 2>nul
if %errorlevel%==0 (
  echo.
  echo Ollama found. For LLM-written explanations run once:   ollama pull gemma2:9b
  echo Without it the app uses rule-based explanations.
) else (
  echo [INFO] Ollama not installed - explanations will be rule-based. https://ollama.com/download
)
echo.
echo Setup complete. Start the app with:  run_app.bat
endlocal
