from pathlib import Path
from templates.ext_build import build_module
import platform, json

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "build" / "artifacts" / "reference"

flags = (["/O1","/std:c++17","/EHsc"]
         if platform.system()=="Windows"
         else ["-O1","-std=c++17"])
meta = build_module("reference_cpp", ROOT/"benchmarks"/"reference_cpp.cpp", flags, ART)
meta["type"]="reference"
print(json.dumps(meta, indent=2))
