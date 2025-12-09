# Build Status Command

Check the build status of all ternary engine modules.

## Task

1. **Check Build Artifacts**
   Look in `build/artifacts/` for:
   - `standard/latest/` - Main SIMD module
   - `dense243/latest/` - Dense243 module
   - Timestamps of latest builds

2. **Verify Module Imports**
   Test that modules can be imported:
   ```python
   import ternary_simd_engine
   import ternary_dense243_module
   ```

3. **Check for .pyd Files**
   Look in project root for compiled extensions:
   - `ternary_simd_engine*.pyd`
   - `ternary_dense243_module*.pyd`

4. **Report Build Age**
   Calculate time since last build for each module

5. **Identify Missing/Outdated Builds**
   - Modules that haven't been built
   - Builds older than 7 days
   - Source files modified after last build

## Output Format

```markdown
## Build Status Report
**Date:** [current date]
**Platform:** Windows x64

### Module Status
| Module | Status | Last Built | Location |
|--------|--------|------------|----------|
| ternary_simd_engine | OK/MISSING/OUTDATED | [date] | [path] |
| ternary_dense243_module | OK/MISSING/OUTDATED | [date] | [path] |
| ternary_tritnet_gemm | OK/MISSING/OUTDATED | [date] | [path] |

### Import Verification
- [x] ternary_simd_engine: Imports successfully
- [ ] ternary_dense243_module: Import failed

### Recommended Actions
[List rebuild commands if needed]

```bash
# Rebuild all modules
python build/build_all.py

# Rebuild specific module
python build/build.py
python build/build_dense243.py
```
```
