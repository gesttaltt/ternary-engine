# Legacy Code Archive

This directory contains code that has been deprecated or removed from active development.

## Contents

### ~~dense243_broken/~~ (RESTORED - 2025-11-23)
**Status:** ✅ Restored to `ternary_engine/experimental/dense243/`

**Why Restored:**
- All tests passing (10/10) - functionality was never broken
- Critical bug fix completed (2025-10-29)
- Performance validated: Pack 0.25ns, Unpack 0.91ns
- TritNet integration planned (neural network-based operations)

**New Location:** `ternary_engine/experimental/dense243/`
**Module:** `ternary_dense243_module` (separate from main engine)

---

## What Was Removed (Not Archived)

### ternary_profiler.h (Deleted 2025-11-22)
**Why:** Never integrated, pure overhead

**Rationale:**
- Defined profiler hooks for VTune/NVTX/Perfetto
- Marked as "ROADMAP FEATURE - NOT YET INTEGRATED"
- Never called from any code
- Added 286 lines of unused complexity
- No tests, no usage, no value until GPU port exists

**If You Need Profiling:**
- Use standard profiling tools (perf, VTune, Nsight) directly
- No custom hooks needed for CPU code
- Reconsider when/if GPU acceleration is added

---

## Archive Policy

Code is archived here instead of deleted when:
1. It has tests or documentation value
2. Future use cases might emerge
3. It represents significant past effort

Code is deleted outright when:
1. Never integrated or used
2. Pure overhead with no tests
3. Adds only complexity, no value
