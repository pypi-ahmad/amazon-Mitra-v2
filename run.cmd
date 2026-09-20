@echo off
setlocal
cd /d "%~dp0"
if not exist .env (
  copy /Y .env.example .env >nul
  start /wait notepad.exe .env
  exit /b 1
)
if not defined HF_TOKEN (
  echo HF_TOKEN is not set in the Windows user environment.
  echo Set it, open a new terminal, and run this launcher again.
  exit /b 1
)
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
if not exist vendor\mitra-finetune\.git (
  git clone https://huggingface.co/autogluon/mitra-finetune vendor\mitra-finetune
  if errorlevel 1 exit /b 1
)
git -C vendor\mitra-finetune fetch --depth 1 origin b4701e8148dc33b00ed15d7086ff59816957cde4
if errorlevel 1 exit /b 1
git -C vendor\mitra-finetune checkout --detach b4701e8148dc33b00ed15d7086ff59816957cde4
if errorlevel 1 exit /b 1
.venv\Scripts\pip install .\vendor\mitra-finetune
if errorlevel 1 exit /b 1
for /f %%P in ('powershell.exe -NoProfile -Command "Get-NetTCPConnection -LocalPort 8541 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique"') do (
  echo Closing existing process %%P on port 8541...
  taskkill /PID %%P /F >nul 2>&1
)
echo Starting Glass Box at http://localhost:8541
.venv\Scripts\streamlit run app.py --server.port 8541
