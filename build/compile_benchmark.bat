@echo off
REM Compile C# LoadProfilingBenchmark

echo Compiling LoadProfilingBenchmark.cs...

C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe ^
    /nologo ^
    /out:dist\NavierLib\LoadProfilingBenchmark.exe ^
    /platform:x64 ^
    /optimize+ ^
    benchmarks\LoadProfilingBenchmark.cs

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Compilation successful!
    echo    Output: dist\NavierLib\LoadProfilingBenchmark.exe
) else (
    echo.
    echo ✗ Compilation failed with error code %ERRORLEVEL%
)

exit /b %ERRORLEVEL%
