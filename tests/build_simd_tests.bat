@echo off
REM build_simd_tests.bat - Build SIMD correctness tests on Windows

echo ========================================
echo   Building SIMD Correctness Tests
echo ========================================
echo.

set PROJECT_ROOT=%~dp0..
set TEST_SOURCE=%PROJECT_ROOT%\tests\test_simd_correctness.cpp
set OUTPUT=%PROJECT_ROOT%\tests\test_simd_correctness.exe

echo Configuration:
echo   Source:  %TEST_SOURCE%
echo   Output:  %OUTPUT%
echo   Compiler: MSVC (cl.exe)
echo.

echo Compiling with AVX2 support...
cl /EHsc /std:c++17 /O2 /arch:AVX2 /I"%PROJECT_ROOT%" "%TEST_SOURCE%" /Fe:"%OUTPUT%" /nologo

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Compilation FAILED
    exit /b 1
)

echo.
echo ========================================
echo   ✓ Compilation Successful
echo ========================================
echo.
echo Run tests with: %OUTPUT%
echo.
