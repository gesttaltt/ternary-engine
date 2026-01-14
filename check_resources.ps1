Write-Host "=== TOP RAM CONSUMERS ===" -ForegroundColor Yellow
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 20 ProcessName, Id, @{N='RAM_MB';E={[math]::Round($_.WorkingSet64/1MB)}} | Format-Table -AutoSize

Write-Host "`n=== TOP CPU CONSUMERS ===" -ForegroundColor Yellow
Get-Process | Sort-Object CPU -Descending | Select-Object -First 15 ProcessName, Id, @{N='CPU_Sec';E={[math]::Round($_.CPU,1)}} | Format-Table -AutoSize

Write-Host "`n=== PYTHON PROCESSES ===" -ForegroundColor Yellow
Get-Process python* -ErrorAction SilentlyContinue | Select-Object Id, @{N='RAM_MB';E={[math]::Round($_.WorkingSet64/1MB)}}, @{N='CPU_Sec';E={[math]::Round($_.CPU,1)}} | Format-Table -AutoSize

Write-Host "`n=== SYSTEM MEMORY ===" -ForegroundColor Yellow
$os = Get-CimInstance Win32_OperatingSystem
$totalRAM = [math]::Round($os.TotalVisibleMemorySize/1MB, 1)
$freeRAM = [math]::Round($os.FreePhysicalMemory/1MB, 1)
$usedRAM = $totalRAM - $freeRAM
Write-Host "Total: ${totalRAM} GB | Used: ${usedRAM} GB | Free: ${freeRAM} GB"
