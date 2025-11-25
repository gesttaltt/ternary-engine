"""
system_load_monitor.py - System Load Monitoring for Reproducible Benchmarks

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Monitors system load from external applications to provide context for benchmark
results. Tracks CPU, memory, and identifies common high-load applications.

Key monitored applications:
- Browsers: Chrome, Firefox, Edge, Opera, Brave
- Development: Docker, VS Code, Visual Studio, PyCharm, IntelliJ
- Communication: Slack, Discord, Teams, Zoom
- Cloud/Sync: Google Drive, OneDrive, Dropbox, iCloud
- System: Antivirus, Windows Defender, indexing services

USAGE:
    from benchmarks.utils.system_load_monitor import SystemLoadMonitor

    monitor = SystemLoadMonitor()
    snapshot = monitor.get_snapshot()
    print(f"System load: {snapshot['load_classification']}")
    print(f"Active high-load apps: {snapshot['high_load_processes']}")
"""

import time
import platform
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed. Install with: pip install psutil")


# Known high-load process patterns and their categories
HIGH_LOAD_PROCESSES = {
    # Browsers (typically 2-15% CPU each, high memory)
    'browsers': [
        'chrome', 'firefox', 'msedge', 'opera', 'brave', 'safari',
        'chromium', 'vivaldi', 'arc'
    ],
    # Docker and containers
    'docker': [
        'docker', 'dockerd', 'containerd', 'docker desktop', 'com.docker',
        'vpnkit', 'wsl'
    ],
    # Development tools
    'development': [
        'code', 'vscode', 'devenv', 'pycharm', 'idea', 'webstorm',
        'android studio', 'xcode', 'atom', 'sublime', 'notepad++',
        'cursor', 'windsurf', 'claude'
    ],
    # Cloud/Sync services (Google Drive, OneDrive, etc.)
    'cloud_sync': [
        'googledrivesync', 'google drive', 'onedrive', 'dropbox',
        'icloud', 'backblaze', 'crashplan', 'carbonite',
        'syncthing', 'resilio', 'nextcloud'
    ],
    # Communication apps
    'communication': [
        'slack', 'discord', 'teams', 'zoom', 'webex', 'skype',
        'telegram', 'whatsapp', 'signal', 'element'
    ],
    # Antivirus/Security
    'security': [
        'msmpeng', 'mssense', 'avast', 'avg', 'norton', 'mcafee',
        'kaspersky', 'bitdefender', 'malwarebytes', 'defender'
    ],
    # Media/Entertainment
    'media': [
        'spotify', 'itunes', 'vlc', 'obs', 'streamlabs',
        'nvidia broadcast', 'geforce experience'
    ],
    # System services
    'system': [
        'searchindexer', 'searchprotocolhost', 'wsearch',
        'windows update', 'trustedinstaller', 'tiworker'
    ]
}


@dataclass
class ProcessInfo:
    """Information about a running process"""
    name: str
    pid: int
    cpu_percent: float
    memory_mb: float
    category: str = "other"


@dataclass
class SystemSnapshot:
    """Complete system load snapshot"""
    timestamp: str
    cpu_percent_total: float
    cpu_percent_per_core: List[float]
    cpu_count_logical: int
    cpu_count_physical: int
    memory_total_gb: float
    memory_used_gb: float
    memory_percent: float
    high_load_processes: List[Dict]
    load_by_category: Dict[str, float]
    load_classification: str  # 'low', 'medium', 'high', 'very_high'
    load_score: float  # 0-100 normalized score
    recommendations: List[str]
    platform_info: Dict


class SystemLoadMonitor:
    """
    Monitors system load for benchmark reproducibility.

    Provides:
    - Real-time CPU/memory monitoring
    - Detection of known high-load applications
    - Load classification for result context
    - Recommendations for cleaner benchmark runs
    """

    def __init__(self, sample_interval: float = 0.5):
        """
        Args:
            sample_interval: Seconds between CPU samples for accurate measurement
        """
        if not HAS_PSUTIL:
            raise RuntimeError("psutil is required. Install with: pip install psutil")

        self.sample_interval = sample_interval
        self._process_cache = {}
        self._last_sample_time = 0

    def _categorize_process(self, name: str) -> str:
        """Categorize a process by its name"""
        name_lower = name.lower()

        for category, patterns in HIGH_LOAD_PROCESSES.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return category

        return "other"

    def _get_process_list(self, min_cpu: float = 0.1) -> List[ProcessInfo]:
        """Get list of processes with significant CPU usage"""
        processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            try:
                info = proc.info
                cpu = info.get('cpu_percent', 0) or 0

                if cpu >= min_cpu:
                    mem_info = info.get('memory_info')
                    mem_mb = mem_info.rss / (1024 * 1024) if mem_info else 0

                    processes.append(ProcessInfo(
                        name=info['name'],
                        pid=info['pid'],
                        cpu_percent=cpu,
                        memory_mb=mem_mb,
                        category=self._categorize_process(info['name'])
                    ))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return sorted(processes, key=lambda p: p.cpu_percent, reverse=True)

    def _calculate_load_score(
        self,
        cpu_percent: float,
        memory_percent: float,
        high_load_count: int
    ) -> float:
        """
        Calculate normalized load score (0-100)

        Weights:
        - CPU usage: 50%
        - Memory usage: 30%
        - High-load process count: 20%
        """
        cpu_score = min(cpu_percent, 100)
        mem_score = min(memory_percent, 100)
        process_score = min(high_load_count * 10, 100)  # 10 points per high-load app

        return (cpu_score * 0.5) + (mem_score * 0.3) + (process_score * 0.2)

    def _classify_load(self, load_score: float) -> str:
        """Classify load level based on score"""
        if load_score < 20:
            return "low"
        elif load_score < 40:
            return "medium"
        elif load_score < 60:
            return "high"
        else:
            return "very_high"

    def _generate_recommendations(
        self,
        load_classification: str,
        high_load_procs: List[ProcessInfo]
    ) -> List[str]:
        """Generate recommendations for cleaner benchmark runs"""
        recommendations = []

        if load_classification in ('high', 'very_high'):
            recommendations.append("Consider closing non-essential applications for more accurate benchmarks")

        # Category-specific recommendations
        categories_found = set(p.category for p in high_load_procs)

        if 'browsers' in categories_found:
            browser_cpu = sum(p.cpu_percent for p in high_load_procs if p.category == 'browsers')
            recommendations.append(f"Browsers using {browser_cpu:.1f}% CPU - close tabs or browsers for cleaner results")

        if 'docker' in categories_found:
            recommendations.append("Docker Desktop running - container workloads may affect benchmarks")

        if 'cloud_sync' in categories_found:
            recommendations.append("Cloud sync active (Google Drive/OneDrive) - disk I/O may affect results")

        if 'communication' in categories_found:
            recommendations.append("Communication apps running - may cause intermittent CPU spikes")

        if 'security' in categories_found:
            security_cpu = sum(p.cpu_percent for p in high_load_procs if p.category == 'security')
            if security_cpu > 5:
                recommendations.append(f"Antivirus/security using {security_cpu:.1f}% CPU - scans may affect benchmarks")

        return recommendations

    def get_snapshot(self) -> Dict:
        """
        Get complete system load snapshot.

        Returns:
            Dictionary with all system metrics and load classification
        """
        # Initial CPU sample (psutil needs interval for accurate reading)
        psutil.cpu_percent(interval=None, percpu=True)
        time.sleep(self.sample_interval)

        # Get CPU metrics
        cpu_percent_per_core = psutil.cpu_percent(interval=None, percpu=True)
        cpu_percent_total = sum(cpu_percent_per_core) / len(cpu_percent_per_core)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)

        # Get memory metrics
        mem = psutil.virtual_memory()
        memory_total_gb = mem.total / (1024 ** 3)
        memory_used_gb = mem.used / (1024 ** 3)
        memory_percent = mem.percent

        # Get process list
        processes = self._get_process_list(min_cpu=0.5)

        # Filter high-load processes (non-other category with >1% CPU or >500MB RAM)
        high_load_procs = [
            p for p in processes
            if p.category != "other" and (p.cpu_percent > 1.0 or p.memory_mb > 500)
        ]

        # Calculate load by category
        load_by_category = {}
        for category in HIGH_LOAD_PROCESSES.keys():
            category_procs = [p for p in processes if p.category == category]
            load_by_category[category] = sum(p.cpu_percent for p in category_procs)

        # Calculate load score and classification
        load_score = self._calculate_load_score(
            cpu_percent_total,
            memory_percent,
            len(high_load_procs)
        )
        load_classification = self._classify_load(load_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(load_classification, high_load_procs)

        # Platform info
        platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }

        snapshot = SystemSnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_percent_total=round(cpu_percent_total, 2),
            cpu_percent_per_core=[round(c, 2) for c in cpu_percent_per_core],
            cpu_count_logical=cpu_count_logical,
            cpu_count_physical=cpu_count_physical,
            memory_total_gb=round(memory_total_gb, 2),
            memory_used_gb=round(memory_used_gb, 2),
            memory_percent=round(memory_percent, 2),
            high_load_processes=[asdict(p) for p in high_load_procs[:10]],  # Top 10
            load_by_category={k: round(v, 2) for k, v in load_by_category.items()},
            load_classification=load_classification,
            load_score=round(load_score, 2),
            recommendations=recommendations,
            platform_info=platform_info
        )

        return asdict(snapshot)

    def get_load_summary(self) -> str:
        """Get a one-line load summary for benchmark headers"""
        snapshot = self.get_snapshot()

        high_load_names = [p['name'] for p in snapshot['high_load_processes'][:5]]
        apps_str = ', '.join(high_load_names) if high_load_names else 'none'

        return (
            f"Load: {snapshot['load_classification'].upper()} "
            f"(score: {snapshot['load_score']:.0f}/100) | "
            f"CPU: {snapshot['cpu_percent_total']:.1f}% | "
            f"RAM: {snapshot['memory_percent']:.1f}% | "
            f"High-load apps: {apps_str}"
        )

    def monitor_during(
        self,
        duration_seconds: float,
        sample_interval: float = 1.0
    ) -> Dict:
        """
        Monitor system load over a time period.

        Args:
            duration_seconds: How long to monitor
            sample_interval: Seconds between samples

        Returns:
            Statistics including min, max, mean, and all samples
        """
        samples = []
        start_time = time.time()

        while (time.time() - start_time) < duration_seconds:
            snapshot = self.get_snapshot()
            samples.append({
                'timestamp': snapshot['timestamp'],
                'cpu_percent': snapshot['cpu_percent_total'],
                'memory_percent': snapshot['memory_percent'],
                'load_score': snapshot['load_score'],
                'load_classification': snapshot['load_classification']
            })
            time.sleep(sample_interval)

        if not samples:
            return {'error': 'No samples collected'}

        cpu_values = [s['cpu_percent'] for s in samples]
        load_scores = [s['load_score'] for s in samples]

        return {
            'duration_seconds': duration_seconds,
            'sample_count': len(samples),
            'cpu_stats': {
                'min': round(min(cpu_values), 2),
                'max': round(max(cpu_values), 2),
                'mean': round(sum(cpu_values) / len(cpu_values), 2),
                'variance': round(
                    sum((x - sum(cpu_values)/len(cpu_values))**2 for x in cpu_values) / len(cpu_values),
                    2
                )
            },
            'load_score_stats': {
                'min': round(min(load_scores), 2),
                'max': round(max(load_scores), 2),
                'mean': round(sum(load_scores) / len(load_scores), 2)
            },
            'classifications': {
                c: sum(1 for s in samples if s['load_classification'] == c)
                for c in ['low', 'medium', 'high', 'very_high']
            },
            'samples': samples
        }


class LoadAwareBenchmark:
    """
    Wrapper for running benchmarks with load context.

    Captures system load before, during, and after benchmark runs
    to provide context for reproducibility.
    """

    def __init__(self):
        self.monitor = SystemLoadMonitor()

    def run_with_context(
        self,
        benchmark_fn,
        name: str = "Benchmark",
        pre_sample_seconds: float = 2.0,
        warmup_fn=None
    ) -> Dict:
        """
        Run a benchmark with full load context.

        Args:
            benchmark_fn: Function that runs the benchmark and returns results
            name: Name of the benchmark
            pre_sample_seconds: How long to sample load before benchmark
            warmup_fn: Optional warmup function to run first

        Returns:
            Results with load context
        """
        print(f"\n{'='*70}")
        print(f"  {name} (Load-Aware)")
        print('='*70)

        # Pre-benchmark load snapshot
        print("\nCapturing pre-benchmark system state...")
        pre_snapshot = self.monitor.get_snapshot()
        print(f"  {self.monitor.get_load_summary()}")

        if pre_snapshot['load_classification'] in ('high', 'very_high'):
            print("\n  WARNING: High system load detected!")
            for rec in pre_snapshot['recommendations']:
                print(f"    - {rec}")

        # Optional warmup
        if warmup_fn:
            print("\nRunning warmup...")
            warmup_fn()

        # Run benchmark
        print(f"\nRunning {name}...")
        start_time = time.time()
        benchmark_results = benchmark_fn()
        elapsed = time.time() - start_time

        # Post-benchmark load snapshot
        print("\nCapturing post-benchmark system state...")
        post_snapshot = self.monitor.get_snapshot()

        # Calculate load stability
        load_change = post_snapshot['load_score'] - pre_snapshot['load_score']
        load_stable = abs(load_change) < 10

        return {
            'name': name,
            'elapsed_seconds': round(elapsed, 3),
            'benchmark_results': benchmark_results,
            'load_context': {
                'pre_benchmark': {
                    'cpu_percent': pre_snapshot['cpu_percent_total'],
                    'memory_percent': pre_snapshot['memory_percent'],
                    'load_score': pre_snapshot['load_score'],
                    'load_classification': pre_snapshot['load_classification'],
                    'high_load_apps': [p['name'] for p in pre_snapshot['high_load_processes']]
                },
                'post_benchmark': {
                    'cpu_percent': post_snapshot['cpu_percent_total'],
                    'memory_percent': post_snapshot['memory_percent'],
                    'load_score': post_snapshot['load_score'],
                    'load_classification': post_snapshot['load_classification']
                },
                'load_change': round(load_change, 2),
                'load_stable': load_stable,
                'recommendations': pre_snapshot['recommendations']
            },
            'reproducibility_notes': self._generate_reproducibility_notes(
                pre_snapshot, post_snapshot, load_stable
            )
        }

    def _generate_reproducibility_notes(
        self,
        pre: Dict,
        post: Dict,
        stable: bool
    ) -> List[str]:
        """Generate notes about result reproducibility"""
        notes = []

        # Load level impact
        if pre['load_classification'] == 'low':
            notes.append("GOOD: Low system load - results should be reproducible")
        elif pre['load_classification'] == 'medium':
            notes.append("OK: Medium system load - results may vary +/- 10%")
        elif pre['load_classification'] == 'high':
            notes.append("WARNING: High system load - results may vary +/- 25%")
        else:
            notes.append("CAUTION: Very high system load - results may vary significantly")

        # Stability
        if stable:
            notes.append("System load was stable during benchmark")
        else:
            notes.append("System load changed during benchmark - consider re-running")

        # Specific impacts
        if pre['load_by_category'].get('browsers', 0) > 5:
            notes.append(f"Browser activity ({pre['load_by_category']['browsers']:.1f}% CPU) may affect results")

        if pre['load_by_category'].get('docker', 0) > 2:
            notes.append("Docker activity may affect memory and I/O benchmarks")

        return notes


def print_system_report():
    """Print a detailed system load report"""
    monitor = SystemLoadMonitor()

    print("="*70)
    print("  SYSTEM LOAD REPORT")
    print("="*70)

    snapshot = monitor.get_snapshot()

    print(f"\nTimestamp: {snapshot['timestamp']}")
    print(f"\nPlatform: {snapshot['platform_info']['system']} {snapshot['platform_info']['release']}")
    print(f"Processor: {snapshot['platform_info']['processor']}")

    print(f"\n--- CPU ---")
    print(f"Total: {snapshot['cpu_percent_total']:.1f}%")
    print(f"Cores: {snapshot['cpu_count_physical']} physical, {snapshot['cpu_count_logical']} logical")
    print(f"Per-core: {snapshot['cpu_percent_per_core']}")

    print(f"\n--- Memory ---")
    print(f"Total: {snapshot['memory_total_gb']:.1f} GB")
    print(f"Used: {snapshot['memory_used_gb']:.1f} GB ({snapshot['memory_percent']:.1f}%)")

    print(f"\n--- Load Classification ---")
    print(f"Score: {snapshot['load_score']:.0f}/100")
    print(f"Classification: {snapshot['load_classification'].upper()}")

    print(f"\n--- Load by Category ---")
    for category, cpu in snapshot['load_by_category'].items():
        if cpu > 0.5:
            print(f"  {category}: {cpu:.1f}% CPU")

    print(f"\n--- High-Load Processes ---")
    if snapshot['high_load_processes']:
        for proc in snapshot['high_load_processes']:
            print(f"  {proc['name']}: {proc['cpu_percent']:.1f}% CPU, {proc['memory_mb']:.0f} MB ({proc['category']})")
    else:
        print("  None detected")

    if snapshot['recommendations']:
        print(f"\n--- Recommendations ---")
        for rec in snapshot['recommendations']:
            print(f"  - {rec}")

    print("\n" + "="*70)

    return snapshot


if __name__ == "__main__":
    print_system_report()
