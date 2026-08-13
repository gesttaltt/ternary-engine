"""
windows_power.py - Windows Power Monitoring Utilities

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Uses native Windows tools to monitor power consumption:
- PowerShell performance counters
- WMIC battery monitoring
- System power state tracking

Requires: Admin privileges for some counters

NOTE (2026-08-13): This module is not currently imported anywhere in the
repo. benchmarks/python-with-interpreter-overhead/bench_power_efficiency.py
-- its only conceivable caller -- reimplements an independent
WindowsPowerMonitor(PowerMonitor) class from scratch instead. Same
duplication pattern already logged for the TritNet Phase 2a/2b training
pipelines (CLAUDE.md gap #7); left as-is rather than merged, per project
policy of not doing speculative refactors, but documented so a future
session doesn't have to rediscover it.

USAGE: python benchmarks/utils/windows_power.py
OUTPUT: Prints system power info and a 5-second power-sampling test to stdout.
"""

import subprocess
import time
import json
from typing import Dict, List, Optional
import re


class WindowsPowerMonitor:
    """
    Power monitoring using Windows-native tools

    Methods:
    - Battery monitoring (laptops)
    - CPU power estimation
    - System power state
    - Performance counters
    """

    def __init__(self):
        self.samples = []
        self.start_time = 0
        self.is_on_battery = self._check_battery_status()

    def _check_battery_status(self) -> bool:
        """Check if system is on battery power"""
        try:
            result = subprocess.run(
                ['powershell', '-Command',
                 '(Get-WmiObject -Class Win32_Battery).BatteryStatus'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # BatteryStatus: 1 = discharging, 2 = AC, 3 = full charged, etc.
            if result.stdout.strip():
                status = int(result.stdout.strip())
                return status == 1  # Discharging = on battery
            return False
        except Exception as e:
            print(f"Battery check failed: {e}")
            return False

    def get_battery_info(self) -> Optional[Dict]:
        """Get detailed battery information"""
        if not self.is_on_battery:
            return None

        try:
            # Get battery discharge rate (watts)
            cmd = '''
            Get-WmiObject -Class Win32_Battery | Select-Object `
                EstimatedChargeRemaining, `
                EstimatedRunTime, `
                @{Name="DischargeRate";Expression={$_.DesignCapacity * (100 - $_.EstimatedChargeRemaining) / 100}}
            '''

            result = subprocess.run(
                ['powershell', '-Command', cmd],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse output
            lines = result.stdout.strip().split('\n')
            info = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()

            return info

        except Exception as e:
            print(f"Battery info failed: {e}")
            return None

    def get_cpu_power_estimate(self) -> float:
        """
        Estimate CPU power consumption using performance counters

        Returns estimated watts (rough approximation)
        """
        try:
            # Get CPU utilization
            cmd = r'Get-Counter "\Processor(_Total)\% Processor Time" | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue'

            result = subprocess.run(
                ['powershell', '-Command', cmd],
                capture_output=True,
                text=True,
                timeout=5
            )

            cpu_percent = float(result.stdout.strip())

            # Rough estimate: Modern CPUs use 15-45W under load
            # This is a very rough approximation
            base_power = 15  # Idle watts
            max_power = 45   # Full load watts
            estimated_watts = base_power + (max_power - base_power) * (cpu_percent / 100.0)

            return estimated_watts

        except Exception as e:
            print(f"CPU power estimate failed: {e}")
            return 25.0  # Default estimate

    def get_system_power_info(self) -> Dict:
        """Get comprehensive system power information"""
        info = {
            'timestamp': time.time(),
            'on_battery': self.is_on_battery,
            'cpu_power_estimate_watts': self.get_cpu_power_estimate()
        }

        # Add battery info if available
        battery_info = self.get_battery_info()
        if battery_info:
            info['battery'] = battery_info

        # Get power scheme
        try:
            result = subprocess.run(
                ['powercfg', '/getactivescheme'],
                capture_output=True,
                text=True,
                timeout=5
            )
            info['power_scheme'] = result.stdout.strip()
        except:
            pass

        return info

    def start_monitoring(self):
        """Start power monitoring session"""
        self.samples = []
        self.start_time = time.time()
        print(f"Power monitoring started (Battery: {self.is_on_battery})")

    def sample(self):
        """Take a power sample"""
        sample = self.get_system_power_info()
        self.samples.append(sample)
        return sample

    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return statistics"""
        end_time = time.time()
        duration = end_time - self.start_time

        if not self.samples:
            return {
                'duration_seconds': duration,
                'samples': 0,
                'average_power_watts': 0
            }

        # Calculate averages
        avg_cpu_power = sum(s['cpu_power_estimate_watts'] for s in self.samples) / len(self.samples)

        result = {
            'duration_seconds': duration,
            'samples': len(self.samples),
            'average_cpu_power_watts': avg_cpu_power,
            'on_battery': self.is_on_battery,
            'total_samples': self.samples
        }

        return result


class WindowsPowerBenchmark:
    """
    Benchmark power consumption using Windows tools

    Continuously samples power during operation
    """

    def __init__(self, sample_interval: float = 0.5):
        """
        Args:
            sample_interval: Seconds between power samples
        """
        self.monitor = WindowsPowerMonitor()
        self.sample_interval = sample_interval

    def benchmark_operation(
        self,
        operation_fn,
        duration_seconds: float = 10.0,
        name: str = "Operation"
    ) -> Dict:
        """
        Benchmark an operation with power monitoring

        Args:
            operation_fn: Function to benchmark
            duration_seconds: How long to run
            name: Operation name

        Returns:
            Statistics including power consumption
        """
        print(f"\nBenchmarking: {name}")
        print(f"Duration: {duration_seconds}s")
        print(f"Sample interval: {self.sample_interval}s")

        self.monitor.start_monitoring()

        # Run operation and sample power
        start_time = time.time()
        iterations = 0
        last_sample_time = start_time
        # Tracks time spent in operation_fn() only, excluding the blocking
        # PowerShell calls issued by self.monitor.sample() (each can take
        # up to its own 5-10s subprocess timeout). Used for ops_per_sec
        # below instead of the raw wall-clock `elapsed`, which used to
        # include that sampling overhead and would understate throughput
        # any time a sample happened to land mid-loop. Found 2026-08-13.
        operation_time = 0.0

        while (time.time() - start_time) < duration_seconds:
            # Run operation
            op_start = time.time()
            operation_fn()
            operation_time += time.time() - op_start
            iterations += 1

            # Sample power at intervals
            if (time.time() - last_sample_time) >= self.sample_interval:
                self.monitor.sample()
                last_sample_time = time.time()

        elapsed = time.time() - start_time

        # Get final statistics
        power_stats = self.monitor.stop_monitoring()

        # Calculate operation metrics (see operation_time comment above)
        ops_per_sec = iterations / operation_time if operation_time > 0 else 0

        # Energy calculation (very rough)
        avg_power = power_stats.get('average_cpu_power_watts', 25)
        energy_joules = avg_power * elapsed
        ops_per_joule = iterations / energy_joules if energy_joules > 0 else 0

        result = {
            'name': name,
            'duration_seconds': elapsed,
            'iterations': iterations,
            'ops_per_second': ops_per_sec,
            'power_stats': power_stats,
            'energy_joules': energy_joules,
            'ops_per_joule': ops_per_joule,
            'joules_per_op': energy_joules / iterations if iterations > 0 else 0
        }

        print(f"\nResults:")
        print(f"  Iterations: {iterations:,}")
        print(f"  Ops/sec: {ops_per_sec:,.0f}")
        print(f"  Avg power: {avg_power:.2f} W")
        print(f"  Energy: {energy_joules:.2f} J")
        print(f"  Ops/Joule: {ops_per_joule:,.0f}")

        return result


def test_windows_power_monitoring():
    """Test Windows power monitoring capabilities"""
    print("=" * 80)
    print("Windows Power Monitoring Test")
    print("=" * 80)

    monitor = WindowsPowerMonitor()

    print("\nSystem Power Info:")
    info = monitor.get_system_power_info()
    print(json.dumps(info, indent=2))

    print("\nSampling power for 5 seconds...")
    monitor.start_monitoring()

    for i in range(5):
        time.sleep(1)
        sample = monitor.sample()
        print(f"  Sample {i+1}: {sample['cpu_power_estimate_watts']:.2f}W")

    stats = monitor.stop_monitoring()
    print("\nFinal Statistics:")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    test_windows_power_monitoring()
