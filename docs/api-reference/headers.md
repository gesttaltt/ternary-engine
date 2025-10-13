# HEADERS

Using C headers on low-level programming context is an art. We need to identify the **sweet spot** where we do not overengineer abstraction but we leverage simplicity. Header-based modular practice is **excellent** when done with discipline, but it has a few sharp edges to manage in general, and in this particular project.

For example in this particular project, our `.cpp` acts as the **abstraction and execution engine**, bridging the low-level C++ logic (SIMD, OpenMP) with Python via **pybind11**, so it’s the runtime layer that exposes your compiled power to higher-level code. You risk ABI mismatches, symbol duplication, memory leaks, undefined behavior, and painful debugging from inconsistent linkage or data ownership across C/C++ boundaries.

---
## Quick Overview

### **When it *is* best practice to use headers**

Use headers **aggressively** when:

* You rely on **`inline` / `constexpr` / templates** → compile-time execution and optimization.
* You want **zero runtime linkage overhead** (e.g., your LUTs, SIMD intrinsics).
* You’re building a **library with clear layers**:
  *math rules → SIMD engine → Python/FFI bridge*.
* You need cross-platform portability (headers carry the same definitions into every TU).

In modern C++ (C++17–20+), header-driven design is the **de-facto standard** for high-performance libraries (Eigen, fmt, pybind11, stb, etc.).

---

### **When it becomes overkill**

It’s counterproductive when:

* You dump **heavy implementations** (large loops, OpenMP sections) into headers — that inflates compile times and binary size.
* You create **circular includes** or **too-many micro-headers** (maintenance nightmare).
* You need **stable binary linkage (ABI)** — then `.cpp` separation is cleaner.

---

### **Optimal rule of thumb**

> **Header = definition of how the system thinks.**
> **CPP = implementation of how it moves.**

So for the project:

* Headers define **ternary algebra laws, traits, and compile-time LUTs.**
* `.cpp` files execute **SIMD, OpenMP, and Python bindings.**

That keeps your build fast, architecture pure, and runtime deterministic — *the perfect balance between modularity and sanity*.