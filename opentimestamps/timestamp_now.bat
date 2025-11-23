@echo off
REM Quick timestamp creation for Windows
REM This creates a timestamp of all critical files

echo ============================================================
echo Creating OpenTimestamps for Ternary Engine
echo ============================================================
echo.

python timestamp_create.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! Timestamp created.
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo ERROR! Timestamp creation failed.
    echo ============================================================
    exit /b 1
)

pause
