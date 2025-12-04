@echo off
REM build_kernels.bat - Build C++ Native Kernel Benchmarks
REM
REM Usage: Run from Developer Command Prompt for VS 2022
REM        Or run vcvarsall.bat x64 first
REM
REM Copyright 2025 Ternary Engine Contributors

echo ================================================================================
echo   C++ Native Kernel Benchmarks - Build Script
echo ================================================================================
echo.

REM Check for cl.exe
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: cl.exe not found in PATH
    echo.
    echo Please run this script from a Developer Command Prompt for VS 2022
    echo Or run vcvarsall.bat x64 first:
    echo.
    echo   "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    echo.
    exit /b 1
)

REM Create output directory
if not exist bin mkdir bin

REM Set paths
set SRC_DIR=..\..\src
set SOURCE_FILE=bench_kernels.cpp
set OUTPUT_FILE=bin\bench_kernels.exe

echo Compiling %SOURCE_FILE%...
echo.

cl /O2 /arch:AVX2 /std:c++17 /EHsc ^
   /I%SRC_DIR% ^
   /Fe:%OUTPUT_FILE% ^
   %SOURCE_FILE%

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Compilation failed!
    exit /b 1
)

echo.
echo ================================================================================
echo   Build Successful!
echo ================================================================================
echo.
echo Executable: %OUTPUT_FILE%
echo.
echo Usage:
echo   %OUTPUT_FILE%              - Run all benchmarks
echo   %OUTPUT_FILE% --csv        - CSV output
echo.

REM Run if --run argument provided
if "%1"=="--run" (
    echo Running benchmark...
    echo.
    %OUTPUT_FILE% %2 %3 %4
)
