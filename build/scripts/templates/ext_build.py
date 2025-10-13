# templates/ext_build.py
import os, sys, subprocess, json
from pathlib import Path
from datetime import datetime

def build_module(name, source, flags, outdir, extra_link_args=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_dir = outdir / ts
    temp = build_dir / "temp"
    out = build_dir / "output"
    latest = outdir / "latest"
    for d in (temp, out): d.mkdir(parents=True, exist_ok=True)

    setup_code = f"""
from setuptools import setup, Extension
import pybind11, os
ext_modules = [
  Extension(
    '{name}',
    [r'{source}'],
    include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
    language='c++',
    extra_compile_args={flags},
    extra_link_args={extra_link_args or []},
  )
]
setup(name='{name}', version='0.1.0', ext_modules=ext_modules, zip_safe=False)
"""
    setup_temp = temp / "setup_temp.py"
    setup_temp.write_text(setup_code)
    subprocess.run([sys.executable, str(setup_temp), "build_ext",
                    "--build-temp", str(temp), "--build-lib", str(out)],
                   check=True)
    if latest.exists(): import shutil; shutil.rmtree(latest)
    import shutil; shutil.copytree(build_dir, latest)
    return {"name": name, "timestamp": ts, "output": str(out)}
