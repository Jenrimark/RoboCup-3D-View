@echo off
echo Starting RoboCup Detection...
echo Activating conda environment: robocup3d

:: Activate conda environment
call conda activate robocup3d
if %errorlevel% neq 0 (
    echo.
    echo Error: Failed to activate conda environment 'robocup3d'
    echo Please make sure the environment exists or create it first
    echo.
    pause
    exit /b 1
)

echo Environment activated successfully
echo Running detection program...

:: Run the program
python pydect4_2025.py

:: Handle program exit
if %errorlevel% neq 0 (
    echo.
    echo Program failed with error code: %errorlevel%
    pause
)
