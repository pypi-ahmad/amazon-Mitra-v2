@echo off
setlocal
cd /d "%~dp0"
if not exist .env (
  copy /Y .env.example .env >nul
  start /wait notepad.exe .env
)
uv python install 3.13
if not exist .venv\Scripts\python.exe uv venv --python 3.13 --seed .venv
.venv\Scripts\pip install -r requirements.txt
for /f %%P in ('powershell.exe -NoProfile -Command "Get-NetTCPConnection -LocalPort 8541 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique"') do (
  echo Closing existing process %%P on port 8541...
  taskkill /PID %%P /F >nul 2>&1
)
echo Starting Glass Box at http://localhost:8541
.venv\Scripts\streamlit run app.py --server.port 8541
