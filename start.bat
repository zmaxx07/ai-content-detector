@echo off
REM ═══════════════════════════════════════════════════════
REM  AI Content Detection System — Start Script (Windows)
REM  Starts backend (port 8000) + frontend (port 3000)
REM ═══════════════════════════════════════════════════════

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║   AI Content Detection System  v3.0.0           ║
echo ╚══════════════════════════════════════════════════╝
echo.

SET ROOT=%~dp0
SET BACKEND=%ROOT%backend
SET FRONTEND=%ROOT%frontend

REM Check .env
IF NOT EXIST "%BACKEND%\.env" (
  echo WARNING: No .env found. Copying from .env.example...
  COPY "%BACKEND%\.env.example" "%BACKEND%\.env"
  echo Edit backend\.env and add your HUGGINGFACE_TOKEN, then re-run.
  PAUSE
  EXIT /B 1
)

REM Check venv
IF NOT EXIST "%BACKEND%\venv" (
  echo Installing Python virtual environment...
  CD "%BACKEND%"
  python -m venv venv
  CALL venv\Scripts\activate.bat
  pip install -r requirements.txt -q
)

REM Check node_modules
IF NOT EXIST "%FRONTEND%\node_modules" (
  echo Installing frontend dependencies...
  CD "%FRONTEND%"
  CALL npm install --silent
)

echo.
echo Starting Backend  -^> http://localhost:8000
echo Starting Frontend -^> http://localhost:3000
echo Press Ctrl+C in each window to stop
echo.

REM Start backend in new window
START "AI Detector Backend" /D "%BACKEND%" cmd /k "venv\Scripts\activate.bat && python run.py"

REM Start frontend in new window
START "AI Detector Frontend" /D "%FRONTEND%" cmd /k "set BROWSER=none && npm start"

echo Both servers starting in separate windows.
echo Open http://localhost:3000 in your browser.
PAUSE
