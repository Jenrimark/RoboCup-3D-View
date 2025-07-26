@echo off
title RoboCup 3D Detection with Conda Environment

echo.
echo ========================================
echo    RoboCup 3D Detection Launcher
echo ========================================
echo.

:: Check if conda is available
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Conda not found in PATH
    echo Please make sure Anaconda or Miniconda is installed and added to PATH
    echo.
    pause
    exit /b 1
)

echo Conda version:
conda --version
echo.

:: Check if robocup3d environment exists
echo Checking conda environment 'robocup3d'...
conda env list | findstr "robocup3d" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Environment 'robocup3d' not found!
    echo.
    echo Would you like to create the environment now? (Y/N)
    set /p create_env=
    if /i "%create_env%"=="Y" (
        echo.
        echo Creating conda environment 'robocup3d' with Python 3.8...
        conda create -n robocup3d python=3.8 -y
        if %errorlevel% neq 0 (
            echo Failed to create environment
            pause
            exit /b 1
        )
        
        echo.
        echo Installing required packages...
        call conda activate robocup3d
        
        echo Installing PyQt5...
        pip install PyQt5
        
        echo Installing OpenCV...
        pip install opencv-python
        
        echo Installing PyTorch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        
        echo Installing Ultralytics...
        pip install ultralytics
        
        echo Installing other dependencies...
        pip install numpy pyorbbecsdk
        
        echo.
        echo Environment setup completed!
    ) else (
        echo Environment creation cancelled
        pause
        exit /b 1
    )
) else (
    echo Environment 'robocup3d' found!
)

echo.
echo Activating conda environment: robocup3d
call conda activate robocup3d
if %errorlevel% neq 0 (
    echo Error: Failed to activate environment
    pause
    exit /b 1
)

echo Environment activated successfully
echo.

:: Check if main program file exists
if not exist "pydect3_2025.py" (
    echo Error: pydect3_2025.py file not found
    echo Please ensure the startup script is in the same directory as the program file
    echo.
    pause
    exit /b 1
)

:: Check model files
echo Checking model files...
set missing_files=0

if not exist "det300.pt" (
    echo Warning: Missing model file - det300.pt
    set missing_files=1
)

if not exist "yuan0517.pt" (
    echo Warning: Missing model file - yuan0517.pt
    set missing_files=1
)

if not exist "fruit.pt" (
    echo Warning: Missing model file - fruit.pt
    set missing_files=1
)

if %missing_files% equ 1 (
    echo.
    echo Some model files are missing. The program may not work properly.
    echo Continue anyway? (Y/N)
    set /p choice=
    if /i not "%choice%"=="Y" (
        echo Startup cancelled
        pause
        exit /b 1
    )
)

echo.
echo Starting RoboCup 3D Detection Program...
echo The program will start detection automatically after loading
echo You can click "Re-detect" button after detection is completed
echo.

:: Run the program
python pydect3_2025.py

:: Handle program exit
echo.
echo Program exited
pause
