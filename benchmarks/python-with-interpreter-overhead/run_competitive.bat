@echo off
REM Competitive Benchmark Runner for Windows
REM Runs all competitive benchmarks and generates reports

echo ==================================
echo Ternary Competitive Benchmark Suite
echo ==================================
echo.

REM Create results directory structure
if not exist results mkdir results
if not exist results\competitive mkdir results\competitive
if not exist results\quantization mkdir results\quantization
if not exist results\power mkdir results\power
if not exist results\reports mkdir results\reports

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    exit /b 1
)

REM Install dependencies
echo Checking dependencies...
python -m pip install numpy --quiet

REM Run benchmarks
echo.
echo ==================================
echo Running Competitive Benchmarks
echo ==================================
echo.

REM Run main competitive suite
echo [1/3] Running main competitive suite (Phases 1-6)...
python bench_competitive.py --all

if errorlevel 1 (
    echo Error: Benchmark failed
    exit /b 1
)

REM Find latest results file
for /f "delims=" %%i in ('dir /b /o-d results\competitive\competitive_results_*.json 2^>nul') do (
    set LATEST_RESULTS=results\competitive\%%i
    goto :found
)

:found
if not defined LATEST_RESULTS (
    echo Error: No results file found
    exit /b 1
)

echo.
echo Results saved to: %LATEST_RESULTS%

REM Generate reports
echo.
echo [2/3] Generating text report...
python utils\visualization.py %LATEST_RESULTS% results\reports\report.txt

echo.
echo [3/3] Generating HTML report...
python utils\visualization.py %LATEST_RESULTS% results\reports\report.html

echo.
echo ==================================
echo Benchmark Complete!
echo ==================================
echo.
echo Results:
echo   JSON:  %LATEST_RESULTS%
echo   Text:  results\reports\report.txt
echo   HTML:  results\reports\report.html
echo.
echo Next steps:
echo   1. Open results\report.html in browser
echo   2. Check competitive viability checklist
echo   3. Run model quantization (Phase 5) if PyTorch available
echo   4. Run power consumption (Phase 6) if on supported hardware
echo.

pause
