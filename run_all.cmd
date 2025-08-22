@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Optional flag: --no-run to skip launching the app (for smoke tests)
set "SKIP_RUN=0"
if /I "%~1"=="--no-run" set "SKIP_RUN=1"

REM --- Go to script directory ---
cd /d %~dp0

echo [1/7] Locating conda...
set "CONDA_BAT="
for /f "usebackq delims=" %%i in (`where conda.bat 2^>nul`) do (
  set "CONDA_BAT=%%i"
  goto :found_conda
)

REM Fallback typical path
if exist "%UserProfile%\anaconda3\condabin\conda.bat" (
  set "CONDA_BAT=%UserProfile%\anaconda3\condabin\conda.bat"
)

:found_conda
if not defined CONDA_BAT (
  echo ERROR: conda.bat not found on PATH and default location.
  echo Please install Anaconda/Miniconda and ensure conda is on PATH.
  exit /b 1
)
echo Using: %CONDA_BAT%

echo [2/7] Ensuring conda env 'loanrisk' exists...
call "%CONDA_BAT%" run -n loanrisk python -V >nul 2>&1
if not errorlevel 1 goto env_exists
echo Creating environment 'loanrisk' (python=3.10)...
call "%CONDA_BAT%" create -n loanrisk python=3.10 -y || goto :error
:env_exists
echo Environment 'loanrisk' is ready.

echo [3/7] Activating environment 'loanrisk'...
call "%CONDA_BAT%" activate loanrisk || goto :error
python -V || goto :error

echo [4/7] Installing requirements...
pip install -r requirements.txt || goto :error

echo [5/7] Generating large dataset (data\applications_large.csv)...
python generate_data.py || goto :error

echo [6/7] Training model on large dataset...
python train_large.py || goto :error

if "%SKIP_RUN%"=="1" goto :done
echo [7/7] Launching Streamlit app...
echo You can stop the app with Ctrl+C. The server will open at http://localhost:8501
streamlit run app.py

goto :eof

:error
echo.
echo FAILED. See the error above. Make sure conda works in a regular cmd prompt.
exit /b 1

:done
echo Completed setup steps. Skipping app launch due to --no-run.
exit /b 0

