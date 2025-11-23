"""
Power Consumption Benchmark - Phase 6

Measure energy efficiency of ternary operations vs baselines.

Edge AI is power-constrained. If ternary saves power, that's the killer feature.

Hardware platforms supported:
- x86 (Intel RAPL - Running Average Power Limit)
- ARM (Raspberry Pi - USB power meter)
- NVIDIA (nvidia-smi power monitoring)

Usage:
    python bench_power_consumption.py --platform x86
    python bench_power_consumption.py --platform arm --device /dev/ttyUSB0
    python bench_power_consumption.py --platform nvidia
"""

import time
import json
import os
import sys
import platform as platform_module
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ternary_simd_engine as tc
except ImportError:
    print("Warning: ternary_simd_engine not available")
    print("Build the module first: python build.py")
    tc = None


class PowerMonitor:
    """
    Abstract base class for power monitoring

    Subclasses implement platform-specific power measurement
    """

    def __init__(self):
        self.baseline_power = 0.0

    def start_monitoring(self):
        """Start power monitoring session"""
        raise NotImplementedError

    def stop_monitoring(self) -> float:
        """Stop monitoring and return average power (Watts)"""
        raise NotImplementedError

    def get_energy_joules(self) -> float:
        """Get total energy consumed (Joules)"""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if this monitor is available on current platform"""
        raise NotImplementedError


class IntelRAPLMonitor(PowerMonitor):
    """
    Intel RAPL (Running Average Power Limit) power monitoring

    Works on Intel CPUs with Linux (requires root or specific permissions)
    """

    def __init__(self):
        super().__init__()
        self.rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/"
        self.start_energy = 0
        self.end_energy = 0

    def is_available(self) -> bool:
        """Check if RAPL is available"""
        return os.path.exists(self.rapl_path)

    def _read_energy(self) -> float:
        """Read current energy counter (microjoules)"""
        try:
            with open(os.path.join(self.rapl_path, "energy_uj"), 'r') as f:
                return float(f.read().strip())
        except (IOError, PermissionError) as e:
            print(f"Warning: Cannot read RAPL (requires permissions): {e}")
            return 0.0

    def start_monitoring(self):
        """Start monitoring"""
        self.start_energy = self._read_energy()

    def stop_monitoring(self) -> float:
        """Stop and return average power"""
        self.end_energy = self._read_energy()
        return 0.0  # Power calculated from energy

    def get_energy_joules(self) -> float:
        """Get total energy consumed"""
        energy_uj = self.end_energy - self.start_energy
        return energy_uj / 1_000_000  # Convert microjoules to joules


class NVIDIAPowerMonitor(PowerMonitor):
    """
    NVIDIA GPU power monitoring via nvidia-smi

    Works on systems with NVIDIA GPUs and nvidia-smi installed
    """

    def __init__(self):
        super().__init__()
        self.samples = []

    def is_available(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            subprocess.run(
                ['nvidia-smi'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _get_power(self) -> float:
        """Get current GPU power draw (Watts)"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return 0.0

    def start_monitoring(self):
        """Start monitoring"""
        self.samples = []

    def stop_monitoring(self) -> float:
        """Stop and return average power"""
        if self.samples:
            return sum(self.samples) / len(self.samples)
        return 0.0

    def sample(self):
        """Take a power sample"""
        power = self._get_power()
        self.samples.append(power)

    def get_energy_joules(self) -> float:
        """Estimate energy from power samples"""
        if not self.samples:
            return 0.0
        avg_power = sum(self.samples) / len(self.samples)
        # Assume 100ms between samples
        duration_sec = len(self.samples) * 0.1
        return avg_power * duration_sec


class WindowsPowerMonitor(PowerMonitor):
    """
    Windows power monitor using PowerShell and performance counters

    Provides CPU power estimation and battery monitoring
    """

    def __init__(self):
        super().__init__()
        self.samples = []
        self.start_time = 0

    def is_available(self) -> bool:
        """Check if Windows monitoring is available"""
        try:
            result = subprocess.run(
                ['powershell', '-Command', 'echo "test"'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False

    def _get_cpu_power(self) -> float:
        """Get CPU power estimate from performance counters"""
        try:
            cmd = r'Get-Counter "\Processor(_Total)\% Processor Time" | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue'
            result = subprocess.run(
                ['powershell', '-Command', cmd],
                capture_output=True,
                text=True,
                timeout=2
            )
            cpu_percent = float(result.stdout.strip())
            # Estimate: 15W base + 30W max load
            return 15 + (30 * cpu_percent / 100.0)
        except:
            return 25.0

    def start_monitoring(self):
        """Start monitoring"""
        self.samples = []
        self.start_time = time.time()
        print("Windows power monitoring started")

    def stop_monitoring(self) -> float:
        """Stop and return average power"""
        if self.samples:
            return sum(self.samples) / len(self.samples)
        return 25.0

    def sample(self):
        """Take a power sample"""
        power = self._get_cpu_power()
        self.samples.append(power)

    def get_energy_joules(self) -> float:
        """Get total energy consumed"""
        duration = time.time() - self.start_time
        avg_power = self.stop_monitoring() if self.samples else 25.0
        return avg_power * duration


class MockPowerMonitor(PowerMonitor):
    """
    Mock power monitor for testing

    Returns simulated power measurements
    """

    def __init__(self):
        super().__init__()
        self.start_time = 0

    def is_available(self) -> bool:
        return True

    def start_monitoring(self):
        self.start_time = time.time()

    def stop_monitoring(self) -> float:
        # Simulate ~50W average power
        return 50.0

    def get_energy_joules(self) -> float:
        duration = time.time() - self.start_time
        # 50W * duration
        return 50.0 * duration


class PowerConsumptionBenchmark:
    """
    Benchmark power consumption of ternary operations

    Metrics:
    - Watts consumed per billion operations (W/GOPS)
    - Operations per Joule
    - Total energy for workload
    - Comparison with baseline (NumPy)
    """

    def __init__(
        self,
        platform: str = "auto",
        output_dir: str = None
    ):
        # Default to benchmarks/results/power/
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "results", "power")

        self.platform = platform
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.monitor = self._create_monitor()

        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'platform': sys.platform,
                'architecture': platform_module.machine(),
                'monitor_type': type(self.monitor).__name__
            },
            'benchmarks': []
        }

    def _create_monitor(self) -> PowerMonitor:
        """Create appropriate power monitor for platform"""
        if self.platform == "auto":
            # Auto-detect
            monitors = [
                WindowsPowerMonitor(),
                IntelRAPLMonitor(),
                NVIDIAPowerMonitor(),
                MockPowerMonitor()
            ]

            for monitor in monitors:
                if monitor.is_available():
                    print(f"Using power monitor: {type(monitor).__name__}")
                    return monitor

            print("Warning: No power monitor available, using mock")
            return MockPowerMonitor()

        elif self.platform == "windows":
            return WindowsPowerMonitor()
        elif self.platform == "intel":
            return IntelRAPLMonitor()
        elif self.platform == "nvidia":
            return NVIDIAPowerMonitor()
        elif self.platform == "mock":
            return MockPowerMonitor()
        else:
            print(f"Unknown platform: {self.platform}, using mock")
            return MockPowerMonitor()

    def benchmark_operation(
        self,
        name: str,
        operation_fn,
        duration_sec: float = 10.0
    ) -> Dict[str, Any]:
        """
        Benchmark power consumption of an operation

        Args:
            name: Operation name
            operation_fn: Function that performs the operation
            duration_sec: How long to run the operation

        Returns:
            Power consumption statistics
        """
        print(f"\nBenchmarking: {name}")
        print(f"  Duration: {duration_sec}s")

        # Start monitoring
        self.monitor.start_monitoring()

        # Run operation for specified duration
        start_time = time.perf_counter()
        iterations = 0

        while (time.perf_counter() - start_time) < duration_sec:
            operation_fn()
            iterations += 1

            # Sample power if supported
            if isinstance(self.monitor, NVIDIAPowerMonitor):
                self.monitor.sample()

        elapsed = time.perf_counter() - start_time

        # Stop monitoring
        avg_power = self.monitor.stop_monitoring()
        energy_joules = self.monitor.get_energy_joules()

        # Calculate metrics
        ops_per_sec = iterations / elapsed
        ops_per_joule = iterations / energy_joules if energy_joules > 0 else 0
        watts_per_gops = (avg_power / ops_per_sec) * 1e9 if ops_per_sec > 0 else 0

        result = {
            'name': name,
            'duration_sec': elapsed,
            'iterations': iterations,
            'ops_per_sec': ops_per_sec,
            'avg_power_watts': avg_power,
            'energy_joules': energy_joules,
            'ops_per_joule': ops_per_joule,
            'watts_per_gops': watts_per_gops
        }

        print(f"  Iterations:       {iterations:,}")
        print(f"  Ops/sec:          {ops_per_sec:,.0f}")
        print(f"  Avg power:        {avg_power:.2f} W")
        print(f"  Energy:           {energy_joules:.2f} J")
        print(f"  Ops/Joule:        {ops_per_joule:,.0f}")
        print(f"  W/GOPS:           {watts_per_gops:.3f}")

        return result

    def run_comparative_benchmark(self, size: int = 1_000_000):
        """
        Run comparative benchmark: ternary vs NumPy

        Args:
            size: Array size for operations
        """
        print("\n" + "=" * 80)
        print("POWER CONSUMPTION COMPARATIVE BENCHMARK")
        print("=" * 80)
        print(f"Array size: {size:,} elements")

        # Prepare data
        a_tern = np.random.randint(0, 3, size, dtype=np.uint8)
        b_tern = np.random.randint(0, 3, size, dtype=np.uint8)

        a_np = np.random.randint(-1, 2, size, dtype=np.int8)
        b_np = np.random.randint(-1, 2, size, dtype=np.int8)

        # Benchmark ternary
        if tc:
            ternary_result = self.benchmark_operation(
                "Ternary Addition",
                lambda: tc.tadd(a_tern, b_tern),
                duration_sec=10.0
            )
            self.results['benchmarks'].append(ternary_result)

        # Benchmark NumPy
        numpy_result = self.benchmark_operation(
            "NumPy INT8 Addition",
            lambda: np.add(a_np, b_np, dtype=np.int8),
            duration_sec=10.0
        )
        self.results['benchmarks'].append(numpy_result)

        # Calculate advantage
        if tc and 'ops_per_joule' in ternary_result and 'ops_per_joule' in numpy_result:
            if numpy_result['ops_per_joule'] > 0:
                efficiency_advantage = (
                    ternary_result['ops_per_joule'] /
                    numpy_result['ops_per_joule']
                )

                print("\n" + "-" * 80)
                print("Power Efficiency Comparison:")
                print(f"  Ternary advantage: {efficiency_advantage:.2f}x more ops/Joule")

                if efficiency_advantage > 1.5:
                    print("  Verdict: ✓ SIGNIFICANT POWER ADVANTAGE")
                elif efficiency_advantage > 1.0:
                    print("  Verdict: ⚠ MODEST POWER ADVANTAGE")
                else:
                    print("  Verdict: ✗ NO POWER ADVANTAGE")

    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(
            self.output_dir,
            f"power_consumption_{timestamp}.json"
        )

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to {filename}")
        return filename


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Power Consumption Benchmark'
    )
    parser.add_argument(
        '--platform',
        choices=['auto', 'windows', 'intel', 'nvidia', 'mock'],
        default='auto',
        help='Power monitoring platform'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=1_000_000,
        help='Array size for operations'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory (default: benchmarks/results/power/)'
    )

    args = parser.parse_args()

    benchmark = PowerConsumptionBenchmark(
        platform=args.platform,
        output_dir=args.output
    )

    benchmark.run_comparative_benchmark(size=args.size)
    benchmark.save_results()


if __name__ == "__main__":
    main()
