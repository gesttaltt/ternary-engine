@echo off
REM NavierLib Benchmark Console - One-click execution
REM Runs performance comparison between NavierLib and C# baseline

echo ================================================================================
echo NavierLib v1.2 Evaluation - Benchmark Suite
echo ================================================================================
echo.

REM Check if DLL exists
if not exist "..\navierlib.dll" (
    echo ERROR: navierlib.dll not found in parent directory
    echo Please ensure navierlib.dll is in the same folder as README.commercial.md
    pause
    exit /b 1
)

REM Check if executable exists
if not exist "NavierBenchConsole.exe" (
    echo ERROR: NavierBenchConsole.exe not found
    echo Please build the benchmark console first (see BUILD_INSTRUCTIONS.md)
    pause
    exit /b 1
)

REM Copy DLL to current directory for execution
copy /Y "..\navierlib.dll" "navierlib.dll" >nul
copy /Y "..\NavierLib.cs" "NavierLib.cs" >nul 2>&1

echo Starting benchmark...
echo.

REM Run benchmark
NavierBenchConsole.exe

echo.
echo ================================================================================
pause
