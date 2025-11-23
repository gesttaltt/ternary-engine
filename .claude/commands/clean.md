Clean build artifacts and temporary files.

Run the comprehensive cleanup script:
```bash
python build/clean_all.py
```

This will:
- Remove all build artifacts from build/ directory
- Clean compiled Python extensions (.pyd, .so)
- Remove temporary compilation files
- Clean __pycache__ directories
- Remove .pyc and .pyo files

**Selective cleanup options:**

Clean only build artifacts:
```bash
python build/clean_all.py --builds-only
```

Clean only Python cache:
```bash
python build/clean_all.py --cache-only
```

**Manual cleanup** (if script unavailable):
```bash
# Windows
rmdir /s /q build
del *.pyd
del *.exp
del *.lib

# Linux/macOS
rm -rf build/
rm -f *.so
find . -type d -name __pycache__ -exec rm -rf {} +
```

Note: Cleanup is safe - it only removes generated files, not source code.
