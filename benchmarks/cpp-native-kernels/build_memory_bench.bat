@echo off
REM build_memory_bench.bat - Build the Memory Efficiency Benchmark
REM
REM Run from Developer Command Prompt for VS 2022

echo ================================================================================
echo   Memory Efficiency Benchmark - Build Script
echo ================================================================================
echo.

where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: cl.exe not found. Run from Developer Command Prompt.
    exit /b 1
)

if not exist bin mkdir bin

echo Compiling bench_memory_efficiency.cpp...

cl /O2 /arch:AVX2 /std:c++17 /EHsc ^
   /I..\..\src ^
   /I.\include ^
   /Fe:bin\bench_memory.exe ^
   bench_memory_efficiency.cpp

if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b 1
)

echo.
echo Build successful: bin\bench_memory.exe
echo.
echo Usage:
echo   bin\bench_memory.exe              - Full benchmark
echo   bin\bench_memory.exe --quiet      - Minimal output
echo   bin\bench_memory.exe --no-models  - Skip model comparison
echo.

if "%1"=="--run" (
    echo Running benchmark...
    bin\bench_memory.exe
)
