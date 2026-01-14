# Firefox memory analysis
Write-Host "=== FIREFOX PROCESSES BY RAM ===" -ForegroundColor Yellow
Get-Process firefox -ErrorAction SilentlyContinue |
    Select-Object Id,
        @{N='RAM_MB';E={[math]::Round($_.WorkingSet64/1MB)}},
        @{N='CPU_Sec';E={[math]::Round($_.CPU,1)}},
        @{N='Title';E={$_.MainWindowTitle}} |
    Sort-Object RAM_MB -Descending |
    Format-Table -AutoSize

Write-Host "`n=== FIREFOX TOTAL ===" -ForegroundColor Yellow
$total = (Get-Process firefox -EA SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum
Write-Host "Total Firefox RAM: $([math]::Round($total/1MB)) MB ($([math]::Round($total/1GB, 2)) GB)"
Write-Host "Process count: $((Get-Process firefox -EA SilentlyContinue).Count)"

Write-Host "`n=== FIREFOX COMMAND LINES (process types) ===" -ForegroundColor Yellow
Get-CimInstance Win32_Process |
    Where-Object {$_.Name -eq 'firefox.exe'} |
    ForEach-Object {
        $cmd = $_.CommandLine
        $type = if ($cmd -match '-contentproc') {
            if ($cmd -match 'gpu') { 'GPU' }
            elseif ($cmd -match 'socket') { 'Network' }
            elseif ($cmd -match 'rdd') { 'Media Decoder' }
            elseif ($cmd -match 'utility') { 'Utility' }
            else { 'Content/Tab' }
        } else { 'Main/Parent' }

        $proc = Get-Process -Id $_.ProcessId -EA SilentlyContinue
        [PSCustomObject]@{
            PID = $_.ProcessId
            Type = $type
            RAM_MB = if($proc){[math]::Round($proc.WorkingSet64/1MB)}else{0}
        }
    } | Sort-Object RAM_MB -Descending | Format-Table -AutoSize

Write-Host "`nTip: Open 'about:processes' in Firefox to see per-tab memory usage" -ForegroundColor Cyan
