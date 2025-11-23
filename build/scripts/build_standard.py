from pathlib import Path
from templates.ext_build import build_module
import platform, json

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "build" / "artifacts" / "standard"

flags = (["/O2","/GL","/arch:AVX2","/openmp","/std:c++17","/EHsc"]
         if platform.system()=="Windows"
         else ["-O3","-march=native","-fopenmp","-std=c++17"])
link = ["/LTCG"] if platform.system()=="Windows" else ["-flto"]

meta = build_module("ternary_simd_engine", ROOT/"ternary_engine"/"ternary_simd_engine.cpp", flags, ART, link)
meta["type"]="standard"
print(json.dumps(meta, indent=2))
