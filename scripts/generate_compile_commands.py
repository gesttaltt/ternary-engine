"""
generate_compile_commands.py - Generate compile_commands.json for IDE support

This script generates a compilation database (compile_commands.json) that
enables C++ language servers like clangd to provide accurate IntelliSense.

Usage: python scripts/generate_compile_commands.py

Output: compile_commands.json in project root

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""
import json
import platform
from pathlib import Path
from typing import List, Dict


def get_compiler_flags() -> str:
    """Get compiler flags based on platform."""
    system = platform.system()

    if system == "Windows":
        # MSVC flags
        return "/O2 /std:c++17 /arch:AVX2 /EHsc /W4 /wd4244 /wd4267"
    elif system == "Darwin":
        # macOS Clang flags
        return "-O3 -std=c++17 -Wall -Wextra"
    else:
        # Linux GCC/Clang flags
        return "-O3 -std=c++17 -march=haswell -mavx2 -Wall -Wextra"


def get_include_paths(project_root: Path) -> List[str]:
    """Get all include paths for the project."""
    include_dirs = [
        project_root / "src" / "core",
        project_root / "src" / "core" / "algebra",
        project_root / "src" / "core" / "simd",
        project_root / "src" / "core" / "common",
        project_root / "src" / "core" / "config",
        project_root / "src" / "core" / "ffi",
        project_root / "src" / "core" / "packing",
        project_root / "src" / "core" / "profiling",
        project_root / "src" / "engine",
        project_root / "src" / "engine" / "dense243",
    ]

    # Filter to existing directories
    return [str(d) for d in include_dirs if d.exists()]


def get_defines() -> List[str]:
    """Get preprocessor defines."""
    defines = [
        "AVX2_ENABLED",
        "NDEBUG",
    ]

    if platform.system() == "Windows":
        defines.extend(["_WIN32", "_WIN64"])

    return defines


def format_include_flags(include_paths: List[str]) -> str:
    """Format include paths as compiler flags."""
    system = platform.system()

    if system == "Windows":
        return " ".join(f'/I"{path}"' for path in include_paths)
    else:
        return " ".join(f'-I"{path}"' for path in include_paths)


def format_define_flags(defines: List[str]) -> str:
    """Format defines as compiler flags."""
    system = platform.system()

    if system == "Windows":
        return " ".join(f"/D{d}" for d in defines)
    else:
        return " ".join(f"-D{d}" for d in defines)


def generate_compile_commands() -> None:
    """Generate compile_commands.json for C++ IDE support."""
    project_root = Path(__file__).parent.parent.resolve()

    # Get configuration
    compiler_flags = get_compiler_flags()
    include_paths = get_include_paths(project_root)
    defines = get_defines()

    include_flags = format_include_flags(include_paths)
    define_flags = format_define_flags(defines)

    # Determine compiler command
    system = platform.system()
    if system == "Windows":
        compiler = "cl"
        compile_suffix = "/c"
    else:
        compiler = "clang++" if system == "Darwin" else "g++"
        compile_suffix = "-c"

    # Collect all C++ files
    cpp_patterns = ["**/*.cpp", "**/*.cc", "**/*.cxx"]
    header_patterns = ["**/*.h", "**/*.hpp", "**/*.hxx"]

    cpp_files: List[Path] = []
    for pattern in cpp_patterns:
        cpp_files.extend(project_root.glob(f"src/{pattern}"))

    header_files: List[Path] = []
    for pattern in header_patterns:
        header_files.extend(project_root.glob(f"src/{pattern}"))

    # Generate commands
    commands: List[Dict] = []

    all_files = cpp_files + header_files
    for file_path in all_files:
        command = f"{compiler} {compiler_flags} {define_flags} {include_flags} {compile_suffix} {file_path}"

        commands.append({
            "directory": str(project_root),
            "file": str(file_path),
            "command": command
        })

    # Write output
    output_path = project_root / "compile_commands.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(commands, f, indent=2)

    print(f"Generated {output_path}")
    print(f"  - {len(cpp_files)} source files")
    print(f"  - {len(header_files)} header files")
    print(f"  - {len(commands)} total entries")
    print(f"  - Platform: {system}")
    print(f"  - Compiler: {compiler}")


def main() -> None:
    """Main entry point."""
    generate_compile_commands()


if __name__ == "__main__":
    main()
