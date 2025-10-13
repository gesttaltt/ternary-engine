import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "benchmarks" / "results"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

print("Building optimized engine for benchmarks...")
res = subprocess.check_output([sys.executable, "build/scripts/build_standard.py"])
meta = json.loads(res)
meta["benchmark_ready"] = True
meta["commit_hash"] = subprocess.getoutput("git rev-parse --short HEAD")
meta["compiler"] = "clang++ -O3" if sys.platform!="win32" else "MSVC /O2 AVX2"
(BENCH_DIR / f"build_meta_{meta['timestamp']}.json").write_text(json.dumps(meta, indent=2))
print(f"✅ Benchmark build complete: {meta['timestamp']}")
