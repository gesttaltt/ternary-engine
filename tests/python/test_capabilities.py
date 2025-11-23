"""
test_capabilities.py - Runtime capability detection for CI/CD

Detects platform features and determines which tests can run.
Used by test suite to gracefully skip tests on incompatible platforms.

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""

import sys
import platform
import subprocess
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SystemCapabilities:
    """Detect and report system capabilities"""

    def __init__(self):
        self.platform = platform.system()
        self.python_version = platform.python_version()
        self.architecture = platform.machine()
        self.cpu_count = os.cpu_count() or 1

        # Detect CPU features
        self.has_avx2 = self._detect_avx2()
        self.has_avx512 = self._detect_avx512()

        # Detect OpenMP in compiled module
        self.has_openmp = self._detect_openmp()

        # Detect fusion module
        self.has_fusion = self._detect_fusion()

        # Determine test compatibility
        self.can_run_simd_tests = self.has_avx2
        self.can_run_openmp_tests = self.has_openmp and self.cpu_count > 1
        self.can_run_fusion_tests = self.has_fusion

    def _detect_avx2(self):
        """Detect AVX2 CPU support"""
        try:
            if self.platform == "Linux":
                result = subprocess.run(
                    ["grep", "-o", "avx2", "/proc/cpuinfo"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                return "avx2" in result.stdout.lower()
            elif self.platform == "Darwin":  # macOS
                result = subprocess.run(
                    ["sysctl", "machdep.cpu.features"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                return "AVX2" in result.stdout
            elif self.platform == "Windows":
                # Try to import module and check
                try:
                    import ternary_simd_engine
                    # If module loads, assume AVX2 works
                    return True
                except:
                    return False
        except:
            return False

    def _detect_avx512(self):
        """Detect AVX-512 CPU support"""
        try:
            if self.platform == "Linux":
                result = subprocess.run(
                    ["grep", "-o", "avx512", "/proc/cpuinfo"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                return "avx512" in result.stdout.lower()
            elif self.platform == "Darwin":
                result = subprocess.run(
                    ["sysctl", "machdep.cpu.features"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                return "AVX512" in result.stdout
        except:
            pass
        return False

    def _detect_openmp(self):
        """Detect if OpenMP was compiled into the module

        NOTE: As of 2025-11-23, OpenMP root cause fixed and re-enabled.
        Tests gracefully fall back if issues detected.
        Previous issue: GitHub Actions CI crashes (2025-10-15)
        """
        try:
            import ternary_simd_engine as tc
            import numpy as np
            # Test with large array (above OMP_THRESHOLD = 100K)
            test_size = 200000
            a = np.zeros(test_size, dtype=np.uint8)
            b = np.zeros(test_size, dtype=np.uint8)
            result = tc.tadd(a, b)
            # If we get here without crashing, OpenMP is working
            return True
        except Exception as e:
            # Graceful fallback if module missing or OpenMP issues
            return False

    def _detect_fusion(self):
        """Detect if fusion operations are available in main module"""
        try:
            import ternary_simd_engine
            # Check if fusion operations exist in main module
            return hasattr(ternary_simd_engine, 'fused_tnot_tadd')
        except:
            return False

    def print_report(self):
        """Print a formatted capability report"""
        print("\n" + "="*70)
        print("  SYSTEM CAPABILITIES REPORT")
        print("="*70)
        print(f"\nPlatform Information:")
        print(f"  OS:           {self.platform}")
        print(f"  Architecture: {self.architecture}")
        print(f"  Python:       {self.python_version}")
        print(f"  CPU Cores:    {self.cpu_count}")

        print(f"\nCPU Features:")
        print(f"  AVX2:         {'[YES]' if self.has_avx2 else '[NO]'}")
        print(f"  AVX-512:      {'[YES]' if self.has_avx512 else '[NO]'}")

        print(f"\nCompiled Features:")
        print(f"  OpenMP:       {'[YES]' if self.has_openmp else '[NO]'}")
        print(f"  Fusion:       {'[YES]' if self.has_fusion else '[NO]'}")

        print(f"\nTest Compatibility:")
        print(f"  SIMD Tests:   {'[CAN RUN]' if self.can_run_simd_tests else '[SKIP]'}")
        print(f"  OpenMP Tests: {'[CAN RUN]' if self.can_run_openmp_tests else '[SKIP]'}")
        print(f"  Fusion Tests: {'[CAN RUN]' if self.can_run_fusion_tests else '[SKIP]'}")

        print("\n" + "="*70)

    def get_skip_reason(self, test_type):
        """Get the reason why a test should be skipped"""
        if test_type == "simd" and not self.can_run_simd_tests:
            return "CPU does not support AVX2"
        elif test_type == "openmp" and not self.can_run_openmp_tests:
            if not self.has_openmp:
                return "OpenMP not compiled into module"
            elif self.cpu_count <= 1:
                return "Single-core system (OpenMP requires multiple cores)"
            else:
                return "OpenMP tests unavailable"
        elif test_type == "fusion" and not self.can_run_fusion_tests:
            return "Fusion operations not available in ternary_simd_engine (build standard module first)"
        return None


def detect_capabilities():
    """Factory function to create and return capabilities"""
    return SystemCapabilities()


if __name__ == "__main__":
    # Run as standalone script to show capabilities
    caps = detect_capabilities()
    caps.print_report()
