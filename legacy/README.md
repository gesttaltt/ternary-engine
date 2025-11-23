# Legacy Code Archive

This directory contains code that has been deprecated or removed from active development.

## Contents

### dense243_broken/
**Status:** Broken - Removed from experimental (2025-11-22)

**Original Purpose:** T5-Dense243 high-density encoding (5 trits/byte, 95.3% density)

**Why Removed:**
- Documented as broken in CHANGELOG.md and COMMERCIABILITY_ASSESSMENT.md
- Failed to deliver promised performance benefits
- Added complexity without validated use cases
- Tests exist but functionality is non-working

**If You Need This:**
- Tests are in `tests/test_dense243.cpp`
- Original headers preserved for reference
- Reconsider only if compelling use case emerges with proper validation

**Alternative:** Use standard 2-bit encoding (4 trits/byte) which is production-validated

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
