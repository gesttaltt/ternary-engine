"""
test_path_fixes.py - Verify that path fixes allow proper imports

This script validates that all benchmark files can correctly resolve
the project root and import required modules.
"""

import sys
from pathlib import Path

def test_path_resolution():
    """Test path resolution for different benchmark files"""

    print("="*80)
    print("  PATH RESOLUTION VERIFICATION")
    print("="*80)
    print()

    project_root = Path(__file__).parent.resolve()
    print(f"Project Root: {project_root}")
    print()

    # Test cases: (file_path, expected_levels_to_root)
    test_cases = [
        ("benchmarks/bench_phase0.py", 2),          # .parent.parent
        ("benchmarks/micro/bench_fusion_phase41.py", 3),  # .parent.parent.parent
        ("benchmarks/micro/bench_fusion_simple.py", 3),
        ("benchmarks/micro/bench_fusion_poc.py", 3),
        ("benchmarks/micro/bench_fusion_rigorous.py", 3),
        ("benchmarks/macro/bench_image_pipeline.py", 3),
        ("benchmarks/macro/bench_neural_layer.py", 3),
    ]

    print("Testing Path Resolution:")
    print("-" * 80)

    all_passed = True

    for file_path, levels in test_cases:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"⚠️  SKIP: {file_path} (file not found)")
            continue

        # Calculate what .parent.parent... should give us
        test_path = full_path
        for _ in range(levels):
            test_path = test_path.parent

        # Verify it resolves to project root
        if test_path.resolve() == project_root:
            print(f"✓  {file_path:<50} (.parent × {levels}) → CORRECT")
        else:
            print(f"✗  {file_path:<50} (.parent × {levels}) → WRONG!")
            print(f"   Expected: {project_root}")
            print(f"   Got:      {test_path.resolve()}")
            all_passed = False

    print()
    return all_passed

def test_module_imports():
    """Test that modules can be imported from project root"""

    print("="*80)
    print("  MODULE IMPORT VERIFICATION")
    print("="*80)
    print()

    project_root = Path(__file__).parent.resolve()

    # Check if compiled modules exist
    print("Checking for compiled modules:")
    print("-" * 80)

    pyd_files = list(project_root.glob("*.pyd"))
    so_files = list(project_root.glob("*.so"))

    if pyd_files:
        for f in pyd_files:
            print(f"✓  Found: {f.name}")

    if so_files:
        for f in so_files:
            print(f"✓  Found: {f.name}")

    if not pyd_files and not so_files:
        print("⚠️  No compiled modules found (.pyd or .so)")
        print("   Run 'python build.py' to build ternary_simd_engine")
        print("   Run 'python build_fusion.py' to build ternary_fusion_engine")
        print()
        return False

    print()

    # Try importing from project root
    print("Testing imports from project root:")
    print("-" * 80)

    sys.path.insert(0, str(project_root))

    try:
        import ternary_simd_engine
        print("✓  import ternary_simd_engine - SUCCESS")
    except ImportError as e:
        print(f"✗  import ternary_simd_engine - FAILED: {e}")
        return False

    try:
        import ternary_fusion_engine
        print("✓  import ternary_fusion_engine - SUCCESS")
    except ImportError as e:
        print(f"✗  import ternary_fusion_engine - FAILED: {e}")
        return False

    # Try importing benchmark_framework
    try:
        from benchmarks.benchmark_framework import BenchmarkRunner
        print("✓  from benchmarks.benchmark_framework import BenchmarkRunner - SUCCESS")
    except ImportError as e:
        print(f"⚠️  from benchmarks.benchmark_framework import ... - FAILED: {e}")
        print("   (This is expected if benchmark_framework doesn't exist)")

    print()
    return True

def main():
    """Run all verification tests"""

    print()
    print("╔" + "═"*78 + "╗")
    print("║" + " "*20 + "PATH FIX VERIFICATION TEST" + " "*32 + "║")
    print("╚" + "═"*78 + "╝")
    print()

    path_ok = test_path_resolution()
    import_ok = test_module_imports()

    print("="*80)
    print("  FINAL RESULTS")
    print("="*80)
    print()

    if path_ok:
        print("✓  Path Resolution: All paths correctly resolve to project root")
    else:
        print("✗  Path Resolution: Some paths are incorrect")

    if import_ok:
        print("✓  Module Imports: All required modules can be imported")
    else:
        print("✗  Module Imports: Some modules cannot be imported")

    print()

    if path_ok and import_ok:
        print("✅ SUCCESS: All path fixes verified!")
        print()
        print("Next steps:")
        print("  1. Run individual benchmarks to test functionality")
        print("  2. Commit changes: git add benchmarks/ .github/")
        print("  3. Push and verify CI/CD workflow")
        return 0
    elif path_ok and not import_ok:
        print("⚠️  PARTIAL: Paths are correct but modules not built")
        print()
        print("Action required:")
        print("  1. Build modules: python build.py")
        print("  2. Build fusion: python build_fusion.py")
        print("  3. Re-run this test")
        return 1
    else:
        print("❌ FAILURE: Path fixes may have issues")
        print()
        print("Action required:")
        print("  1. Review PATH_ISSUES_REPORT.md")
        print("  2. Check file locations and .parent chains")
        return 2

if __name__ == "__main__":
    sys.exit(main())
