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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # fixed 2026-08-12: was 1 dirname() short of repo root

try:
    import ternary_simd_engine as tc
except ImportError:
    print("Warning: ternary_simd_engine not available")
    print("Build the module first: python build.py")
    tc = None


class PowerMonitor:
    #: Subclasses that fabricate rather than measure set this True.
    IS_SIMULATED = False
    #: Which physical device this monitor observes. The workloads in this
    #: file are CPU workloads (ternary_simd_engine's AVX2 kernels and NumPy
    #: baselines), so a monitor whose MEASURES != 'cpu' cannot substantiate
    #: a claim about them, no matter how real its readings are.
    MEASURES = 'cpu'

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
        self.max_energy_range_uj = self._read_max_energy_range()

    def _read_max_energy_range(self) -> float:
        """Read this domain's counter wrap point (microjoules), for
        wraparound correction in get_energy_joules(). Falls back to 0
        (wraparound correction becomes a no-op) if unreadable -- this
        file is typically world-readable even when energy_uj isn't, but
        don't assume."""
        try:
            with open(os.path.join(self.rapl_path, "max_energy_range_uj"), 'r') as f:
                return float(f.read().strip())
        except (IOError, PermissionError, ValueError):
            return 0.0

    def is_available(self) -> bool:
        """Check if RAPL is genuinely usable: the energy_uj file must exist
        AND actually be readable by this process, not just the directory.

        Found 2026-08-22: `energy_uj` is root-only (`-r--------`) on a
        stock Linux install unless a udev rule grants group/world read
        access -- the common case for an unprivileged user, not a sandbox
        quirk. The prior check (`os.path.exists(self.rapl_path)`, the
        directory) is True regardless of that file's permissions, so this
        monitor would report itself "available", then silently read 0.0J
        for every measurement (via _read_energy()'s own PermissionError
        fallback below) behind one easy-to-miss warning per call -- the
        same silent-degrade shape already found and fixed elsewhere in
        this project's benchmarks/ and models/ (see .claude/CLAUDE.md
        Critical Gaps). Attempting a real read here lets the auto-detect
        cascade in PowerConsumptionBenchmark._create_monitor() correctly
        fall through to NVIDIAPowerMonitor/MockPowerMonitor instead.
        """
        energy_path = os.path.join(self.rapl_path, "energy_uj")
        if not os.path.exists(energy_path):
            return False
        try:
            with open(energy_path, 'r') as f:
                f.read()
            return True
        except (IOError, PermissionError):
            return False

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
        """Get total energy consumed.

        RAPL's energy_uj is a wrapping counter (wraps at
        max_energy_range_uj, ~65.5kJ on this hardware -- confirmed via
        max_energy_range_uj, not assumed) -- a long enough or high-enough-
        power measurement window can wrap mid-benchmark, making
        end_energy < start_energy and producing a nonsensical negative
        Joules reading without this correction. Not reachable at this
        module's default duration_sec=10.0 on typical desktop/laptop
        power draw, but cheap to guard unconditionally rather than rely
        on that always holding for every caller/duration.
        """
        energy_uj = self.end_energy - self.start_energy
        if energy_uj < 0 and self.max_energy_range_uj > 0:
            energy_uj += self.max_energy_range_uj
        return energy_uj / 1_000_000  # Convert microjoules to joules


class NVIDIAPowerMonitor(PowerMonitor):
    """
    NVIDIA GPU power monitoring via nvidia-smi

    Works on systems with NVIDIA GPUs and nvidia-smi installed
    """

    def __init__(self):
        super().__init__()
        self.samples = []

    MEASURES = 'gpu'

    def is_available(self) -> bool:
        """True when nvidia-smi is present. NOTE: this monitor is excluded
        from _create_monitor()'s auto-detect chain (2026-08-29) -- the
        exclusion is there, not here, because "is nvidia-smi available" is
        a genuine question worth answering honestly for an explicit
        --platform nvidia run.

        Why it was excluded: it used to sit ahead of MockPowerMonitor in
        that chain, so on this host
        -- CPU ternary kernels, an NVIDIA GPU present, and RAPL's energy_uj
        root-only -- `--platform auto` therefore selected GPU power
        monitoring for a purely CPU workload and reported ~11.3 W of *idle
        GPU* draw as the energy cost of AVX2 ternary operations, computed a
        "1.02x ternary advantage" from it, and exited 0 with
        is_simulated=False.

        That is more insidious than the mock-power fallback fixed alongside
        it: nothing is fabricated, so no simulation marker would ever catch
        it -- the readings are real measurements of the wrong device. Use
        `--platform nvidia` explicitly, and only for a GPU workload; main()
        still exits 2 on the device mismatch if the workload is CPU.
        """
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

    #: Marks this monitor as fabricating its numbers. Checked before any
    #: result is reported or saved -- see PowerConsumptionBenchmark.
    IS_SIMULATED = True

    def is_available(self) -> bool:
        """Deliberately NOT auto-selectable (fixed 2026-08-29).

        This previously returned True and sat last in _create_monitor()'s
        auto-detect chain, so on any machine where RAPL's energy_uj is
        root-only -- the default on modern kernels, and the exact situation
        on this host -- `--platform auto` silently selected a monitor that
        fabricates 50 W, computed a real-looking ops/joule ratio from it,
        and saved it to JSON with no marker of any kind, exiting 0.

        That is the same silent-fallback class already fixed twice in this
        repo (bench_competitive.py's mock (a+b)%3 arithmetic;
        test_falsification.py substituting NumPy for the real engine), and
        it would corrupt commercial-viability criterion 4 specifically.
        Selecting simulated power now requires an explicit
        `--platform mock`.
        """
        return False

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
                'monitor_type': type(self.monitor).__name__,
                # Explicit, machine-checkable marker: True means the numbers
                # in this file are FABRICATED, not measured. Added 2026-08-29
                # alongside closing the silent auto-fallback to
                # MockPowerMonitor -- previously nothing in a saved result
                # distinguished a real run from a simulated one.
                'is_simulated': bool(getattr(self.monitor, 'IS_SIMULATED', False)),
                'measures_device': getattr(self.monitor, 'MEASURES', 'cpu'),
            },
            'benchmarks': []
        }

    def _create_monitor(self) -> PowerMonitor:
        """Create appropriate power monitor for platform"""
        if self.platform == "auto":
            # Auto-detect
            # CPU monitors only: every workload in this file runs on the
            # CPU. NVIDIAPowerMonitor (GPU) and MockPowerMonitor (fabricated)
            # are reachable only via an explicit --platform, for the reasons
            # in their is_available() docstrings.
            monitors = [
                WindowsPowerMonitor(),
                IntelRAPLMonitor(),
            ]

            for monitor in monitors:
                if monitor.is_available():
                    print(f"Using power monitor: {type(monitor).__name__}")
                    return monitor
            # falls through to the explicit error below

            print("=" * 70)
            print("ERROR: No real power monitor is available on this machine.")
            print("  Intel RAPL is the usual path here; its energy_uj is")
            print("  root-only by default (post-PLATYPUS mitigation). Either:")
            print("    sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj \\")
            print("                   /sys/class/powercap/intel-rapl:0:0/energy_uj")
            print("  or run this benchmark under sudo.")
            print("  Refusing to fall back to simulated power, which would")
            print("  produce fabricated numbers indistinguishable from real")
            print("  ones. Pass --platform mock if you explicitly want that.")
            print("=" * 70)
            raise SystemExit(2)

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

    # A simulated run must be impossible to mistake for a real one, including
    # by a caller that only checks the exit code (exit 2, distinct from
    # success 0). Mirrors the fix already applied to test_falsification.py.
    if getattr(benchmark.monitor, "MEASURES", "cpu") != "cpu":
        print()
        print("!" * 70)
        print("!! DEVICE MISMATCH -- the selected monitor measures "
              f"{getattr(benchmark.monitor, 'MEASURES', '?').upper()},")
        print("!! but every workload in this benchmark runs on the CPU.")
        print("!! These readings are real but describe the wrong device.")
        print("!! Do not cite this run for commercial-viability criterion 4.")
        print("!" * 70)
        return 2

    if getattr(benchmark.monitor, "IS_SIMULATED", False):
        print()
        print("!" * 70)
        print("!! SIMULATED POWER -- THESE NUMBERS ARE FABRICATED, NOT MEASURED")
        print("!! MockPowerMonitor reports a fixed 50 W regardless of workload.")
        print("!! Do not cite this run for commercial-viability criterion 4.")
        print("!" * 70)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
