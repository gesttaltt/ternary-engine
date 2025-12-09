# Development Tooling & MCP Server Guide

**Doc-Type:** Developer Guide · Version 1.0 · Updated 2025-12-09 · Author Ternary Engine Team

Comprehensive guide for setting up MCP (Model Context Protocol) servers and development tooling for the Ternary Engine project. This documentation covers installation, configuration, and best practices for an optimized development workflow.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [MCP Servers](#mcp-servers)
   - [Memory MCP](#1-memory-mcp-server)
   - [GitHub MCP](#2-github-mcp-server)
   - [Sequential Thinking MCP](#3-sequential-thinking-mcp-server)
   - [Filesystem MCP](#4-filesystem-mcp-server)
4. [Python Tooling](#python-tooling)
   - [Package Management](#package-management-pyprojecttoml)
   - [Ruff Linter](#ruff-linter--formatter)
   - [Mypy Type Checker](#mypy-type-checker)
   - [Pytest](#pytest-testing-framework)
5. [C++ Tooling](#c-tooling)
   - [Clang-Format](#clang-format)
   - [Clang-Tidy](#clang-tidy)
   - [Cppcheck](#cppcheck)
   - [Compile Commands](#compile-commands-for-ide-support)
6. [VSCode Configuration](#vscode-configuration)
   - [Settings](#vscode-settings)
   - [Extensions](#recommended-extensions)
   - [Tasks](#build-tasks)
7. [Performance Analysis Tools](#performance-analysis-tools)
   - [Intel VTune](#intel-vtune-profiler)
   - [py-spy](#py-spy-python-profiler)
   - [pytest-benchmark](#pytest-benchmark)
8. [Documentation Tools](#documentation-tools)
   - [MkDocs](#mkdocs)
   - [Doxygen](#doxygen)
9. [Claude Code Enhancements](#claude-code-enhancements)
   - [Custom Slash Commands](#custom-slash-commands)
10. [Quick Setup](#quick-setup)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This guide establishes a professional development environment for the Ternary Engine project, focusing on:

- **MCP Servers**: Extend Claude Code capabilities with persistent memory, GitHub integration, and enhanced reasoning
- **Code Quality**: Automated linting, formatting, and type checking for Python and C++
- **Performance Analysis**: Profiling tools for optimization work
- **IDE Integration**: VSCode configuration for seamless development

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Development Environment                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Memory MCP  │  │  GitHub MCP  │  │ Sequential   │          │
│  │  (Context)   │  │  (Issues/PR) │  │  Thinking    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           │                                     │
│                    ┌──────▼───────┐                             │
│                    │  Claude Code │                             │
│                    └──────┬───────┘                             │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                   │
│         │                 │                 │                   │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐          │
│  │    Python    │  │     C++      │  │   VSCode     │          │
│  │  Ruff/Mypy   │  │ Clang-Format │  │   Config     │          │
│  │   Pytest     │  │ Clang-Tidy   │  │  Extensions  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before installing the tooling, ensure you have:

### Required Software

| Software | Version | Purpose | Installation |
|----------|---------|---------|--------------|
| **Node.js** | 18+ | MCP server runtime | [nodejs.org](https://nodejs.org/) |
| **Python** | 3.7+ | Core development | [python.org](https://python.org/) |
| **MSVC** | 2017+ | C++ compilation | Visual Studio Build Tools |
| **Git** | 2.30+ | Version control | [git-scm.com](https://git-scm.com/) |

### Verification Commands

```powershell
# Verify Node.js
node --version
# Expected: v18.x.x or higher

# Verify npm
npm --version
# Expected: 9.x.x or higher

# Verify Python
python --version
# Expected: Python 3.7+

# Verify Git
git --version
# Expected: git version 2.30+

# Verify MSVC (Windows)
cl
# Expected: Microsoft (R) C/C++ Optimizing Compiler
```

---

## MCP Servers

MCP (Model Context Protocol) servers extend Claude Code with additional capabilities. They run as separate processes and communicate via the MCP protocol.

### Configuration Location

MCP servers are configured in Claude Code's settings file:

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

For Claude Code CLI, use: `~/.claude/settings.json`

---

### 1. Memory MCP Server

**Purpose**: Persistent memory across Claude Code sessions. Store benchmark baselines, phase completion status, and project context.

#### Installation

```powershell
# Install globally (recommended for frequent use)
npm install -g @modelcontextprotocol/server-memory

# Or use npx (downloads on demand)
# No installation needed - configured in settings
```

#### Configuration

Add to your Claude settings file:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

For global installation:

```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-server-memory"
    }
  }
}
```

#### Usage Examples

Once configured, you can ask Claude Code to:

```
"Remember that the Phase 4.1 fusion baseline is 35.2 Gops/s"
"What was the last benchmark result I saved?"
"Store the current test status: 65/65 passing on Windows x64"
"Recall the TritNet training accuracy from last session"
```

#### Use Cases for Ternary Engine

| Use Case | Example Memory Entry |
|----------|---------------------|
| Benchmark baselines | `phase4_fusion_baseline: 35.2 Gops/s (2025-12-09)` |
| Validation status | `windows_x64_validated: true, tests: 65/65` |
| TritNet progress | `tritnet_tnot_accuracy: 98.7%, epochs: 150` |
| Known issues | `gemm_performance_gap: 0.37 vs 20-30 Gops/s target` |

---

### 2. GitHub MCP Server

**Purpose**: Direct GitHub integration for issues, pull requests, releases, and repository management.

#### Installation

```powershell
# Install globally
npm install -g @modelcontextprotocol/server-github
```

#### GitHub Token Setup

1. Go to [GitHub Settings > Developer Settings > Personal Access Tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Select scopes:
   - `repo` (full repository access)
   - `read:org` (if using organization repos)
   - `read:user` (for user info)
4. Copy the generated token

#### Configuration

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

**Security Note**: For better security, use environment variables:

```powershell
# Set environment variable (PowerShell)
[Environment]::SetEnvironmentVariable("GITHUB_TOKEN", "ghp_xxx", "User")
```

Then configure:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

#### Usage Examples

```
"Create an issue for the GEMM performance gap"
"List all open issues in ternary-engine"
"Create a PR for the current branch"
"What are the recent commits on main?"
"Add a label 'performance' to issue #42"
```

#### Use Cases for Ternary Engine

| Use Case | Command |
|----------|---------|
| Track gaps | Create issues for each of the 12 documented gaps |
| Phase tracking | Label issues by phase (phase-1, phase-2a, etc.) |
| Release management | Create releases with benchmark validation |
| Code review | Create PRs with performance comparison |

---

### 3. Sequential Thinking MCP Server

**Purpose**: Enhanced reasoning for complex optimization decisions. Provides structured thinking chains for architectural choices.

#### Installation

```powershell
npm install -g @modelcontextprotocol/server-sequential-thinking
```

#### Configuration

```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

#### Usage Examples

```
"Think through the tradeoffs between LUT-based and neural network-based ternary arithmetic"
"Analyze step-by-step why GEMM performance is 54x slower than target"
"Design an optimization strategy for the fusion operations"
```

#### Use Cases for Ternary Engine

| Use Case | Why Sequential Thinking Helps |
|----------|------------------------------|
| SIMD optimization | Analyze vectorization strategies systematically |
| TritNet architecture | Evaluate hidden layer sizes, activation functions |
| Fusion patterns | Determine which operation combinations benefit most |
| Performance debugging | Trace through cache behavior, memory access patterns |

---

### 4. Filesystem MCP Server

**Purpose**: Enhanced filesystem operations, directory watching, and artifact management.

#### Installation

```powershell
npm install -g @modelcontextprotocol/server-filesystem
```

#### Configuration

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\Alejandro\\Documents\\Ivan\\Work\\ternary-engine"
      ]
    }
  }
}
```

**Note**: You can specify multiple allowed directories:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\Alejandro\\Documents\\Ivan\\Work\\ternary-engine",
        "C:\\Users\\Alejandro\\Documents\\Ivan\\Work\\other-project"
      ]
    }
  }
}
```

#### Usage Examples

```
"Watch the build/artifacts directory for new builds"
"Find all benchmark results from the last week"
"List the most recently modified files in src/core"
```

---

### Complete MCP Configuration

Here's the complete configuration with all recommended MCP servers:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxx"
      }
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\Alejandro\\Documents\\Ivan\\Work\\ternary-engine"
      ]
    }
  }
}
```

---

## Python Tooling

### Package Management (pyproject.toml)

Modern Python projects use `pyproject.toml` for configuration. This file replaces `setup.py`, `setup.cfg`, `requirements.txt`, and tool-specific config files.

#### Installation

No installation needed - `pyproject.toml` is a configuration file. However, you need pip 21.3+ for full support:

```powershell
python -m pip install --upgrade pip
```

#### Configuration

See the `pyproject.toml` file in the project root (created by this setup).

#### Usage

```powershell
# Install the project in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev]"           # Development tools
pip install -e ".[tritnet]"       # TritNet/PyTorch
pip install -e ".[dev,tritnet]"   # Both
```

---

### Ruff (Linter & Formatter)

**Purpose**: Extremely fast Python linter and formatter. Replaces Flake8, isort, Black, and more.

#### Installation

```powershell
# Via pip
pip install ruff

# Or via pipx (isolated installation)
pipx install ruff
```

#### Configuration

Already configured in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py37"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "C90"]
ignore = ["E501"]  # Line length handled separately

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

#### Usage

```powershell
# Check for issues
ruff check .

# Check and fix automatically
ruff check --fix .

# Format code
ruff format .

# Check specific file
ruff check src/engine/bindings_core_ops.cpp
```

#### VSCode Integration

Install the "Ruff" extension (`charliermarsh.ruff`) for automatic formatting on save.

---

### Mypy (Type Checker)

**Purpose**: Static type checking for Python. Catches type errors before runtime.

#### Installation

```powershell
pip install mypy
```

#### Configuration

Already configured in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.7"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
exclude = [
    "build/",
    "models/tritnet/src/generate_truth_tables.py",
]
```

#### Usage

```powershell
# Check entire project
mypy .

# Check specific file
mypy tests/python/test_phase0.py

# Check with strict mode
mypy --strict models/tritnet/src/
```

#### VSCode Integration

Install the "Mypy Type Checker" extension (`ms-python.mypy-type-checker`).

---

### Pytest (Testing Framework)

**Purpose**: Modern Python testing framework with fixtures, parameterization, and plugins.

#### Installation

```powershell
pip install pytest pytest-benchmark pytest-cov
```

#### Configuration

Already configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests/python"]
python_files = "test_*.py"
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow",
    "benchmark: marks benchmark tests",
]
```

#### Usage

```powershell
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/python/test_phase0.py

# Run tests matching pattern
pytest -k "test_add"

# Run with coverage
pytest --cov=src --cov-report=html

# Run benchmarks
pytest --benchmark-only
```

---

## C++ Tooling

### Clang-Format

**Purpose**: Automatic C++ code formatting for consistent style.

#### Installation

**Windows (via LLVM):**

```powershell
# Using Chocolatey
choco install llvm

# Or download from: https://releases.llvm.org/
# Add to PATH: C:\Program Files\LLVM\bin
```

**Verify Installation:**

```powershell
clang-format --version
# Expected: clang-format version 15.0.0 or higher
```

#### Configuration

Create `.clang-format` in project root:

```yaml
---
Language: Cpp
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 100
AllowShortFunctionsOnASingleLine: Inline
AllowShortIfStatementsOnASingleLine: Never
AllowShortLoopsOnASingleLine: false
BreakBeforeBraces: Attach
PointerAlignment: Left
SpaceAfterCStyleCast: false
SpaceBeforeParens: ControlStatements
Standard: c++17
---
```

#### Usage

```powershell
# Format single file
clang-format -i src/core/algebra/ternary_algebra.h

# Format all C++ files
Get-ChildItem -Recurse -Include *.h,*.cpp | ForEach-Object { clang-format -i $_.FullName }

# Check formatting without modifying
clang-format --dry-run -Werror src/core/algebra/ternary_algebra.h
```

---

### Clang-Tidy

**Purpose**: C++ static analysis and linting. Catches bugs, style issues, and performance problems.

#### Installation

Included with LLVM installation (see Clang-Format above).

#### Configuration

Create `.clang-tidy` in project root:

```yaml
---
Checks: >
  -*,
  bugprone-*,
  clang-analyzer-*,
  cppcoreguidelines-*,
  modernize-*,
  performance-*,
  readability-*,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers,
  -cppcoreguidelines-avoid-magic-numbers
WarningsAsErrors: ''
HeaderFilterRegex: 'src/.*'
AnalyzeTemporaryDtors: false
FormatStyle: file
CheckOptions:
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.FunctionCase
    value: camelBack
  - key: readability-identifier-naming.VariableCase
    value: lower_case
---
```

#### Usage

```powershell
# Analyze single file
clang-tidy src/core/algebra/ternary_algebra.h

# Analyze with fixes
clang-tidy -fix src/core/algebra/ternary_algebra.h

# Analyze all files (requires compile_commands.json)
clang-tidy -p build src/core/**/*.h
```

---

### Cppcheck

**Purpose**: Static analysis tool for C/C++ focusing on bug detection.

#### Installation

```powershell
# Using Chocolatey
choco install cppcheck

# Or download from: https://cppcheck.sourceforge.io/
```

#### Usage

```powershell
# Check entire project
cppcheck --enable=all --std=c++17 src/

# Check with specific includes
cppcheck --enable=all --std=c++17 -I src/core -I src/engine src/

# Generate XML report
cppcheck --enable=all --xml src/ 2> cppcheck-report.xml
```

---

### Compile Commands (for IDE Support)

**Purpose**: `compile_commands.json` provides compilation flags to IDEs for accurate IntelliSense.

#### Generation Script

Create `scripts/generate_compile_commands.py`:

```python
"""
Generate compile_commands.json for IDE support.

Usage: python scripts/generate_compile_commands.py
"""
import json
from pathlib import Path

def generate_compile_commands():
    """Generate compile_commands.json for C++ IDE support."""

    project_root = Path(__file__).parent.parent
    src_core = project_root / "src" / "core"
    src_engine = project_root / "src" / "engine"

    # Collect all C++ source files
    cpp_files = list(project_root.glob("src/**/*.cpp"))
    header_files = list(project_root.glob("src/**/*.h"))

    # MSVC compile command template
    msvc_flags = "/O2 /std:c++17 /arch:AVX2 /EHsc /W4"
    includes = f"/I{src_core} /I{src_engine} /I{src_engine}/dense243"

    commands = []

    for cpp_file in cpp_files:
        commands.append({
            "directory": str(project_root),
            "file": str(cpp_file),
            "command": f"cl {msvc_flags} {includes} /c {cpp_file}"
        })

    # Also add headers for IDE navigation
    for header_file in header_files:
        commands.append({
            "directory": str(project_root),
            "file": str(header_file),
            "command": f"cl {msvc_flags} {includes} /c {header_file}"
        })

    output_path = project_root / "compile_commands.json"
    with open(output_path, "w") as f:
        json.dump(commands, f, indent=2)

    print(f"Generated {output_path} with {len(commands)} entries")

if __name__ == "__main__":
    generate_compile_commands()
```

#### Usage

```powershell
python scripts/generate_compile_commands.py
```

This generates `compile_commands.json` in the project root, which VSCode's C++ extension uses for IntelliSense.

---

## VSCode Configuration

### VSCode Settings

Create `.vscode/settings.json`:

```json
{
    // Python Configuration
    "python.defaultInterpreterPath": "python",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,

    // Python Formatting (Ruff)
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },

    // C++ Configuration
    "[cpp]": {
        "editor.defaultFormatter": "ms-vscode.cpptools",
        "editor.formatOnSave": true
    },
    "[c]": {
        "editor.defaultFormatter": "ms-vscode.cpptools"
    },

    // C++ IntelliSense
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.cStandard": "c17",
    "C_Cpp.default.intelliSenseMode": "windows-msvc-x64",
    "C_Cpp.default.defines": [
        "_WIN32",
        "_WIN64",
        "AVX2_ENABLED",
        "NDEBUG"
    ],
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/src/core",
        "${workspaceFolder}/src/core/algebra",
        "${workspaceFolder}/src/core/simd",
        "${workspaceFolder}/src/core/common",
        "${workspaceFolder}/src/core/ffi",
        "${workspaceFolder}/src/engine",
        "${workspaceFolder}/src/engine/dense243"
    ],
    "C_Cpp.clang_format_fallbackStyle": "Google",
    "C_Cpp.codeAnalysis.clangTidy.enabled": true,

    // File Associations
    "files.associations": {
        "*.h": "cpp",
        "*.tritnet": "json",
        "array": "cpp",
        "atomic": "cpp",
        "*.tcc": "cpp",
        "functional": "cpp",
        "tuple": "cpp",
        "type_traits": "cpp",
        "utility": "cpp"
    },

    // Editor Settings
    "editor.rulers": [100],
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "editor.trimAutoWhitespace": true,
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,

    // Search Exclusions
    "search.exclude": {
        "**/build/artifacts": true,
        "**/.mypy_cache": true,
        "**/models/datasets": true,
        "**/*.pyd": true,
        "**/*.so": true
    },

    // Terminal
    "terminal.integrated.defaultProfile.windows": "PowerShell",

    // Git
    "git.autofetch": true,
    "git.confirmSync": false
}
```

### Recommended Extensions

Create `.vscode/extensions.json`:

```json
{
    "recommendations": [
        // Python
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.mypy-type-checker",
        "charliermarsh.ruff",

        // C++
        "ms-vscode.cpptools",
        "ms-vscode.cpptools-extension-pack",
        "ms-vscode.cmake-tools",
        "twxs.cmake",
        "jeff-hykin.better-cpp-syntax",

        // Git
        "eamodio.gitlens",
        "mhutchie.git-graph",

        // Markdown
        "yzhang.markdown-all-in-one",
        "bierner.markdown-mermaid",
        "davidanson.vscode-markdownlint",

        // General
        "editorconfig.editorconfig",
        "streetsidesoftware.code-spell-checker",
        "usernamehw.errorlens",
        "gruntfuggly.todo-tree",

        // AI Assistance
        "github.copilot",
        "anthropics.claude-code"
    ],
    "unwantedRecommendations": []
}
```

### Build Tasks

Create `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build: Standard Module",
            "type": "shell",
            "command": "python",
            "args": ["build/build.py"],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$gcc"],
            "presentation": {
                "echo": true,
                "reveal": "always",
                "panel": "shared"
            }
        },
        {
            "label": "Build: Dense243 Module",
            "type": "shell",
            "command": "python",
            "args": ["build/build_dense243.py"],
            "group": "build",
            "problemMatcher": ["$gcc"]
        },
        {
            "label": "Build: All Modules",
            "type": "shell",
            "command": "python",
            "args": ["build/build_all.py"],
            "group": "build",
            "problemMatcher": ["$gcc"]
        },
        {
            "label": "Test: Run All Tests",
            "type": "shell",
            "command": "python",
            "args": ["tests/run_tests.py"],
            "group": {
                "kind": "test",
                "isDefault": true
            },
            "problemMatcher": []
        },
        {
            "label": "Test: Phase 0 Correctness",
            "type": "shell",
            "command": "python",
            "args": ["tests/python/test_phase0.py"],
            "group": "test",
            "problemMatcher": []
        },
        {
            "label": "Benchmark: Core Operations",
            "type": "shell",
            "command": "python",
            "args": ["benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py"],
            "group": "none",
            "problemMatcher": []
        },
        {
            "label": "Benchmark: All Benchmarks",
            "type": "shell",
            "command": "python",
            "args": ["benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py"],
            "group": "none",
            "problemMatcher": []
        },
        {
            "label": "Clean: All Artifacts",
            "type": "shell",
            "command": "python",
            "args": ["build/clean_all.py"],
            "group": "none",
            "problemMatcher": []
        },
        {
            "label": "Lint: Ruff Check",
            "type": "shell",
            "command": "ruff",
            "args": ["check", "."],
            "group": "none",
            "problemMatcher": []
        },
        {
            "label": "Lint: Mypy Type Check",
            "type": "shell",
            "command": "mypy",
            "args": ["."],
            "group": "none",
            "problemMatcher": []
        },
        {
            "label": "Generate: Compile Commands",
            "type": "shell",
            "command": "python",
            "args": ["scripts/generate_compile_commands.py"],
            "group": "none",
            "problemMatcher": []
        }
    ]
}
```

### Launch Configuration

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Python: Run Tests",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/tests/run_tests.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Python: Benchmark Core Ops",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Python: TritNet Training",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/models/tritnet/run_tritnet.py",
            "args": ["--phase", "2a"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

---

## Performance Analysis Tools

### Intel VTune Profiler

**Purpose**: Deep CPU profiling for SIMD optimization, cache analysis, and hotspot detection.

#### Installation

1. Download Intel oneAPI Base Toolkit from [Intel Developer Zone](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html)
2. Run the installer and select "Intel VTune Profiler"
3. Default installation: `C:\Program Files (x86)\Intel\oneAPI\vtune\latest`

#### Setup

```powershell
# Add to PATH (PowerShell profile)
$env:PATH += ";C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64"

# Or run the setup script
& "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

#### Usage

```powershell
# Hotspot analysis
vtune -collect hotspots -result-dir vtune_results python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py

# Memory access analysis
vtune -collect memory-access -result-dir vtune_memory python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py

# View results
vtune-gui vtune_results
```

#### Integration with Ternary Engine

Create `scripts/profile_vtune.py`:

```python
"""
Run VTune profiling on ternary engine benchmarks.

Usage: python scripts/profile_vtune.py [hotspots|memory|uarch]
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_vtune_profile(analysis_type="hotspots"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(f"reports/vtune/{analysis_type}_{timestamp}")

    cmd = [
        "vtune",
        "-collect", analysis_type,
        "-result-dir", str(result_dir),
        "python",
        "benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py"
    ]

    print(f"Running VTune {analysis_type} analysis...")
    subprocess.run(cmd)
    print(f"Results saved to: {result_dir}")

if __name__ == "__main__":
    analysis = sys.argv[1] if len(sys.argv) > 1 else "hotspots"
    run_vtune_profile(analysis)
```

---

### py-spy (Python Profiler)

**Purpose**: Sampling profiler for Python with minimal overhead. Works with native extensions.

#### Installation

```powershell
pip install py-spy
```

#### Usage

```powershell
# Record profile (generates SVG flamegraph)
py-spy record -o profile.svg -- python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py

# Top-like live view
py-spy top -- python benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py

# Dump current stack traces
py-spy dump --pid <python_pid>
```

#### Integration Script

Create `scripts/profile_pyspy.py`:

```python
"""
Run py-spy profiling on ternary engine benchmarks.

Usage: python scripts/profile_pyspy.py
"""
import subprocess
from pathlib import Path
from datetime import datetime

def run_pyspy_profile():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("reports/profiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"flamegraph_{timestamp}.svg"

    cmd = [
        "py-spy", "record",
        "-o", str(output_file),
        "--", "python",
        "benchmarks/python-with-interpreter-overhead/bench_simd_core_ops.py"
    ]

    print("Running py-spy profiler...")
    subprocess.run(cmd)
    print(f"Flamegraph saved to: {output_file}")

if __name__ == "__main__":
    run_pyspy_profile()
```

---

### pytest-benchmark

**Purpose**: Statistical benchmarking framework with automatic calibration and comparison.

#### Installation

```powershell
pip install pytest-benchmark
```

#### Usage

```powershell
# Run benchmarks
pytest tests/ --benchmark-only

# Compare against saved baseline
pytest tests/ --benchmark-compare

# Save baseline
pytest tests/ --benchmark-save=baseline

# Generate histogram
pytest tests/ --benchmark-histogram
```

#### Example Benchmark Test

Create `tests/python/test_benchmark_ops.py`:

```python
"""
Benchmark tests for ternary operations using pytest-benchmark.
"""
import numpy as np
import pytest

# Import after build
try:
    import ternary_simd_engine as engine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


@pytest.mark.skipif(not HAS_ENGINE, reason="Engine not built")
class TestBenchmarkOps:
    """Benchmark tests for core operations."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample ternary data."""
        np.random.seed(42)
        return np.random.randint(-1, 2, size=100000, dtype=np.int8)

    def test_benchmark_tadd(self, benchmark, sample_data):
        """Benchmark ternary addition."""
        a, b = sample_data, np.roll(sample_data, 1)
        result = benchmark(engine.tadd, a, b)
        assert len(result) == len(a)

    def test_benchmark_tmul(self, benchmark, sample_data):
        """Benchmark ternary multiplication."""
        a, b = sample_data, np.roll(sample_data, 1)
        result = benchmark(engine.tmul, a, b)
        assert len(result) == len(a)

    def test_benchmark_tnot(self, benchmark, sample_data):
        """Benchmark ternary negation."""
        result = benchmark(engine.tnot, sample_data)
        assert len(result) == len(sample_data)
```

---

## Documentation Tools

### MkDocs

**Purpose**: Generate a documentation website from Markdown files.

#### Installation

```powershell
pip install mkdocs mkdocs-material mkdocs-mermaid2-plugin
```

#### Configuration

Create `mkdocs.yml` in project root:

```yaml
site_name: Ternary Engine Documentation
site_description: Production-grade balanced ternary arithmetic library
site_author: Ternary Engine Team
repo_url: https://github.com/gesttaltt/ternary-engine

theme:
  name: material
  palette:
    primary: indigo
    accent: indigo
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate
    - search.suggest
    - content.code.copy

plugins:
  - search
  - mermaid2

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed
  - admonition
  - toc:
      permalink: true

nav:
  - Home: README.md
  - Getting Started:
    - Installation: docs/installation.md
    - Quick Start: docs/quickstart.md
  - API Reference:
    - Core Operations: docs/api-reference/source-code-overview.md
    - SIMD Kernels: docs/api-reference/ternary-core-simd.md
    - Error Handling: docs/api-reference/error-handling.md
  - Architecture:
    - Overview: docs/architecture/architecture.md
    - Optimization: docs/architecture/optimization-roadmap.md
  - TritNet:
    - Vision: docs/research/tritnet/TRITNET_VISION.md
    - Roadmap: docs/research/tritnet/TRITNET_ROADMAP.md
  - Contributing: CONTRIBUTING.md
  - Changelog: CHANGELOG.md
```

#### Usage

```powershell
# Serve locally with hot reload
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

---

### Doxygen

**Purpose**: Generate API documentation from C++ source code comments.

#### Installation

```powershell
# Using Chocolatey
choco install doxygen.install graphviz

# Or download from: https://www.doxygen.nl/download.html
```

#### Configuration

Create `Doxyfile` in project root:

```
PROJECT_NAME           = "Ternary Engine"
PROJECT_BRIEF          = "Production-grade balanced ternary arithmetic library"
OUTPUT_DIRECTORY       = docs/api/cpp
INPUT                  = src/core src/engine
FILE_PATTERNS          = *.h *.cpp
RECURSIVE              = YES
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = NO
EXTRACT_STATIC         = YES
GENERATE_HTML          = YES
GENERATE_LATEX         = NO
HAVE_DOT               = YES
DOT_IMAGE_FORMAT       = svg
INTERACTIVE_SVG        = YES
CALL_GRAPH             = YES
CALLER_GRAPH           = YES
SOURCE_BROWSER         = YES
INLINE_SOURCES         = YES
REFERENCED_BY_RELATION = YES
REFERENCES_RELATION    = YES
```

#### Usage

```powershell
# Generate documentation
doxygen Doxyfile

# Open in browser
start docs/api/cpp/html/index.html
```

---

## Claude Code Enhancements

### Custom Slash Commands

Create additional slash commands for common workflows:

#### Performance Analysis Command

Create `.claude/commands/perf-analyze.md`:

```markdown
Analyze the performance of recent benchmark runs:

1. Find the most recent benchmark results in reports/performance/
2. Compare against the documented baselines in CLAUDE.md:
   - Peak throughput: 45.3 Gops/s (fusion)
   - Element-wise: 39.1 Gops/s
   - Average speedup: 8,234x vs Python
3. Identify any regressions (>5% slowdown)
4. Suggest optimization opportunities based on the data
5. Generate a summary report with:
   - Current vs baseline metrics
   - Regression warnings
   - Next steps for optimization

Format the output as a markdown table for easy reading.
```

#### Validation Command

Create `.claude/commands/validate-phase.md`:

```markdown
Validate a development phase for production readiness:

1. Run the test suite: `python tests/run_tests.py`
2. Run performance benchmarks: `python benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py`
3. Compare results against documented baselines
4. Check that documentation is current
5. Verify no regressions in test count or performance

Generate a validation report with:
- Test results (pass/fail count)
- Performance comparison vs baseline
- Documentation status
- Overall GO/NO-GO recommendation

Include the validation date and platform (Windows x64).
```

#### TritNet Status Command

Create `.claude/commands/tritnet-status.md`:

```markdown
Check TritNet development status and next steps:

1. Review trained models in models/tritnet/
2. Check training history files for accuracy metrics
3. Compare against phase targets:
   - Phase 2A (tnot): 100% accuracy target
   - Phase 2B (all ops): 99%+ accuracy target
4. Review TRITNET_ROADMAP.md for current phase
5. Identify blockers or next steps

Report:
- Current phase and status
- Model accuracy metrics
- Dataset availability
- Recommended next actions
```

#### Build Status Command

Create `.claude/commands/build-status.md`:

```markdown
Check the build status of all modules:

1. Check for existing build artifacts in build/artifacts/
2. Verify module imports work:
   - ternary_simd_engine
   - ternary_dense243_module
3. Report build timestamps
4. Identify any missing or outdated builds
5. Suggest rebuild commands if needed

Output a status table showing:
| Module | Status | Last Built | Location |
```

---

## Quick Setup

### Automated Setup Script

Create `scripts/setup_dev_environment.py`:

```python
"""
setup_dev_environment.py - Complete development environment setup

This script installs all required tools and configurations for
developing on the Ternary Engine project.

Usage: python scripts/setup_dev_environment.py [--full]

Options:
    --full    Install optional tools (VTune, MkDocs, Doxygen)
"""
import subprocess
import sys
import json
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command with description."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"Warning: {description} may have failed")
    return result.returncode == 0


def check_prerequisites():
    """Check required software is installed."""
    print("\nChecking prerequisites...")

    checks = [
        ("node --version", "Node.js"),
        ("python --version", "Python"),
        ("git --version", "Git"),
    ]

    all_ok = True
    for cmd, name in checks:
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode == 0:
            version = result.stdout.decode().strip()
            print(f"  [OK] {name}: {version}")
        else:
            print(f"  [MISSING] {name}")
            all_ok = False

    return all_ok


def install_python_tools():
    """Install Python development tools."""
    run_command(
        "python -m pip install --upgrade pip",
        "Upgrading pip"
    )

    run_command(
        "pip install ruff mypy pytest pytest-benchmark pytest-cov py-spy",
        "Installing Python development tools"
    )

    run_command(
        "pip install numpy pybind11 torch",
        "Installing core dependencies"
    )


def install_mcp_servers():
    """Install MCP servers globally."""
    servers = [
        "@modelcontextprotocol/server-memory",
        "@modelcontextprotocol/server-github",
        "@modelcontextprotocol/server-sequential-thinking",
        "@modelcontextprotocol/server-filesystem",
    ]

    for server in servers:
        run_command(
            f"npm install -g {server}",
            f"Installing {server}",
            check=False
        )


def create_vscode_config():
    """Create VSCode configuration files."""
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)

    # Settings already created by this documentation
    print("  VSCode configuration files should be created from documentation")


def build_modules():
    """Build the ternary engine modules."""
    run_command(
        "python build/build.py",
        "Building standard module"
    )

    run_command(
        "python build/build_dense243.py",
        "Building Dense243 module"
    )


def run_tests():
    """Run the test suite to verify installation."""
    run_command(
        "python tests/run_tests.py",
        "Running test suite"
    )


def generate_compile_commands():
    """Generate compile_commands.json for IDE support."""
    script = Path("scripts/generate_compile_commands.py")
    if script.exists():
        run_command(
            f"python {script}",
            "Generating compile_commands.json"
        )


def print_mcp_config():
    """Print MCP configuration for user to add."""
    config = {
        "mcpServers": {
            "memory": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"]
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token-here>"
                }
            },
            "sequential-thinking": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
            }
        }
    }

    print("\n" + "="*60)
    print("  MCP Configuration")
    print("="*60)
    print("\nAdd this to your Claude settings file:")
    print("  Windows: %APPDATA%\\Claude\\claude_desktop_config.json")
    print("  macOS: ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("\n" + json.dumps(config, indent=2))


def main():
    """Main setup routine."""
    print("="*60)
    print("  Ternary Engine Development Environment Setup")
    print("="*60)

    full_install = "--full" in sys.argv

    if not check_prerequisites():
        print("\nPlease install missing prerequisites and try again.")
        sys.exit(1)

    install_python_tools()
    install_mcp_servers()
    build_modules()
    run_tests()
    generate_compile_commands()
    print_mcp_config()

    if full_install:
        run_command("pip install mkdocs mkdocs-material", "Installing MkDocs")
        run_command("choco install doxygen.install -y", "Installing Doxygen", check=False)

    print("\n" + "="*60)
    print("  Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Add MCP configuration to Claude settings (see above)")
    print("  2. Install recommended VSCode extensions")
    print("  3. Run: python build/build.py")
    print("  4. Run: python tests/run_tests.py")


if __name__ == "__main__":
    main()
```

---

## Troubleshooting

### Common Issues

#### MCP Server Won't Start

**Symptom**: Claude Code shows "MCP server failed to start"

**Solutions**:
1. Verify Node.js is installed: `node --version`
2. Check npm is working: `npm --version`
3. Try installing globally: `npm install -g @modelcontextprotocol/server-memory`
4. Check for firewall blocking localhost connections

#### Ruff Not Formatting

**Symptom**: Files not formatted on save

**Solutions**:
1. Ensure Ruff extension is installed
2. Check `settings.json` has correct formatter configured
3. Verify ruff is installed: `ruff --version`
4. Check for syntax errors in the file (Ruff won't format invalid Python)

#### C++ IntelliSense Not Working

**Symptom**: Red squiggles, missing includes, no autocomplete

**Solutions**:
1. Generate `compile_commands.json`: `python scripts/generate_compile_commands.py`
2. Reload VSCode window: `Ctrl+Shift+P` → "Reload Window"
3. Check include paths in `.vscode/settings.json`
4. Verify C/C++ extension is installed

#### Build Fails

**Symptom**: `python build/build.py` fails

**Solutions**:
1. Ensure MSVC is installed with C++ workload
2. Check AVX2 support: `python -c "import platform; print(platform.processor())"`
3. Verify pybind11 is installed: `pip install pybind11`
4. Check Python version matches build target

#### Tests Fail After Fresh Clone

**Symptom**: Import errors when running tests

**Solutions**:
1. Build the module first: `python build/build.py`
2. Ensure NumPy is installed: `pip install numpy`
3. Check the `.pyd` file was created in project root
4. Verify Python architecture matches build (both 64-bit)

### Getting Help

- **Project Issues**: https://github.com/gesttaltt/ternary-engine/issues
- **MCP Documentation**: https://modelcontextprotocol.io/
- **Claude Code Help**: `/help` command or https://github.com/anthropics/claude-code

---

## Appendix: Configuration Files Summary

| File | Purpose | Location |
|------|---------|----------|
| `pyproject.toml` | Python project configuration | Project root |
| `.vscode/settings.json` | VSCode settings | `.vscode/` |
| `.vscode/extensions.json` | Recommended extensions | `.vscode/` |
| `.vscode/tasks.json` | Build/test tasks | `.vscode/` |
| `.vscode/launch.json` | Debug configurations | `.vscode/` |
| `.clang-format` | C++ formatting rules | Project root |
| `.clang-tidy` | C++ linting rules | Project root |
| `compile_commands.json` | C++ compilation database | Project root |
| `mkdocs.yml` | Documentation site config | Project root |
| `Doxyfile` | C++ API docs config | Project root |

---

**Version:** 1.0.0 · **Updated:** 2025-12-09 · **Project:** Ternary Engine
