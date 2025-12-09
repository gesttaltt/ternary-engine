"""
setup_dev_environment.py - Complete development environment setup

This script installs all required tools and configurations for
developing on the Ternary Engine project.

Usage:
    python scripts/setup_dev_environment.py           # Standard setup
    python scripts/setup_dev_environment.py --full    # Full setup with optional tools
    python scripts/setup_dev_environment.py --check   # Check prerequisites only

Options:
    --full    Install optional tools (MkDocs, visualization)
    --check   Only check prerequisites without installing
    --skip-mcp  Skip MCP server installation

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0
"""
import subprocess
import sys
import json
import platform
from pathlib import Path
from typing import List, Tuple, Optional


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")


def print_status(status: str, message: str) -> None:
    """Print a status message."""
    if status == "OK":
        print(f"  {Colors.GREEN}[OK]{Colors.RESET} {message}")
    elif status == "WARN":
        print(f"  {Colors.YELLOW}[WARN]{Colors.RESET} {message}")
    elif status == "FAIL":
        print(f"  {Colors.RED}[FAIL]{Colors.RESET} {message}")
    elif status == "INFO":
        print(f"  {Colors.BLUE}[INFO]{Colors.RESET} {message}")
    else:
        print(f"  [{status}] {message}")


def run_command(
    cmd: List[str],
    description: str,
    check: bool = True,
    capture: bool = False
) -> Tuple[bool, Optional[str]]:
    """Run a command with description."""
    print_status("INFO", description)
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=check
        )
        if capture:
            return True, result.stdout.strip()
        return True, None
    except subprocess.CalledProcessError as e:
        if capture:
            return False, e.stderr
        return False, None
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"


def check_prerequisites() -> bool:
    """Check required software is installed."""
    print_header("Checking Prerequisites")

    checks = [
        (["node", "--version"], "Node.js", "18.0.0"),
        (["npm", "--version"], "npm", "9.0.0"),
        (["python", "--version"], "Python", "3.7.0"),
        (["git", "--version"], "Git", "2.30.0"),
    ]

    all_ok = True
    for cmd, name, min_version in checks:
        success, output = run_command(cmd, f"Checking {name}...", check=False, capture=True)
        if success and output:
            # Extract version number
            version = output.split()[-1].lstrip("v")
            print_status("OK", f"{name}: {version}")
        else:
            print_status("FAIL", f"{name}: Not found")
            all_ok = False

    # Check for MSVC on Windows
    if platform.system() == "Windows":
        success, output = run_command(
            ["where", "cl"],
            "Checking MSVC compiler...",
            check=False,
            capture=True
        )
        if success:
            print_status("OK", "MSVC: Found")
        else:
            print_status("WARN", "MSVC: Not in PATH (may need Visual Studio Developer Command Prompt)")

    return all_ok


def install_python_tools(full: bool = False) -> bool:
    """Install Python development tools."""
    print_header("Installing Python Tools")

    # Upgrade pip first
    run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip..."
    )

    # Core development tools
    core_tools = [
        "ruff",
        "mypy",
        "pytest",
        "pytest-benchmark",
        "pytest-cov",
        "py-spy",
    ]

    success, _ = run_command(
        [sys.executable, "-m", "pip", "install"] + core_tools,
        "Installing development tools (ruff, mypy, pytest)..."
    )

    # Core dependencies
    run_command(
        [sys.executable, "-m", "pip", "install", "numpy", "pybind11"],
        "Installing core dependencies (numpy, pybind11)..."
    )

    # PyTorch for TritNet
    run_command(
        [sys.executable, "-m", "pip", "install", "torch"],
        "Installing PyTorch for TritNet...",
        check=False  # May fail on some systems
    )

    if full:
        # Optional visualization tools
        run_command(
            [sys.executable, "-m", "pip", "install", "matplotlib"],
            "Installing visualization tools...",
            check=False
        )

        # Documentation tools
        run_command(
            [sys.executable, "-m", "pip", "install", "mkdocs", "mkdocs-material"],
            "Installing documentation tools...",
            check=False
        )

    return True


def install_mcp_servers() -> bool:
    """Install MCP servers globally."""
    print_header("Installing MCP Servers")

    servers = [
        ("@modelcontextprotocol/server-memory", "Memory MCP"),
        ("@modelcontextprotocol/server-github", "GitHub MCP"),
        ("@modelcontextprotocol/server-sequential-thinking", "Sequential Thinking MCP"),
        ("@modelcontextprotocol/server-filesystem", "Filesystem MCP"),
    ]

    all_ok = True
    for package, name in servers:
        success, _ = run_command(
            ["npm", "install", "-g", package],
            f"Installing {name}...",
            check=False
        )
        if success:
            print_status("OK", f"{name} installed")
        else:
            print_status("WARN", f"{name} installation failed (may already be installed)")
            all_ok = False

    return all_ok


def build_modules() -> bool:
    """Build the ternary engine modules."""
    print_header("Building Modules")

    project_root = Path(__file__).parent.parent

    # Build standard module
    success, _ = run_command(
        [sys.executable, str(project_root / "build" / "build.py")],
        "Building standard SIMD module..."
    )
    if not success:
        print_status("FAIL", "Standard module build failed")
        return False

    # Build Dense243 module
    success, _ = run_command(
        [sys.executable, str(project_root / "build" / "build_dense243.py")],
        "Building Dense243 module...",
        check=False
    )

    return True


def run_tests() -> bool:
    """Run the test suite to verify installation."""
    print_header("Running Tests")

    project_root = Path(__file__).parent.parent

    success, _ = run_command(
        [sys.executable, str(project_root / "tests" / "run_tests.py")],
        "Running test suite..."
    )

    return success


def generate_compile_commands() -> bool:
    """Generate compile_commands.json for IDE support."""
    print_header("Generating IDE Support Files")

    project_root = Path(__file__).parent.parent
    src_core = project_root / "src" / "core"
    src_engine = project_root / "src" / "engine"

    # Collect all C++ files
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

    for header_file in header_files:
        commands.append({
            "directory": str(project_root),
            "file": str(header_file),
            "command": f"cl {msvc_flags} {includes} /c {header_file}"
        })

    output_path = project_root / "compile_commands.json"
    with open(output_path, "w") as f:
        json.dump(commands, f, indent=2)

    print_status("OK", f"Generated compile_commands.json with {len(commands)} entries")
    return True


def print_mcp_config() -> None:
    """Print MCP configuration for user to add."""
    print_header("MCP Configuration")

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
            },
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    str(Path(__file__).parent.parent.absolute())
                ]
            }
        }
    }

    print("\nAdd this configuration to your Claude settings file:")
    print(f"\n  {Colors.YELLOW}Windows:{Colors.RESET} %APPDATA%\\Claude\\claude_desktop_config.json")
    print(f"  {Colors.YELLOW}macOS:{Colors.RESET}   ~/Library/Application Support/Claude/claude_desktop_config.json")
    print(f"  {Colors.YELLOW}Linux:{Colors.RESET}   ~/.config/Claude/claude_desktop_config.json")

    print(f"\n{Colors.BOLD}Configuration:{Colors.RESET}")
    print(json.dumps(config, indent=2))

    print(f"\n{Colors.YELLOW}Note:{Colors.RESET} Replace <your-token-here> with your GitHub personal access token")
    print("Generate one at: https://github.com/settings/tokens")


def print_next_steps() -> None:
    """Print next steps for the user."""
    print_header("Setup Complete!")

    print(f"""
{Colors.GREEN}Your development environment is ready!{Colors.RESET}

{Colors.BOLD}Next Steps:{Colors.RESET}

  1. {Colors.YELLOW}Configure MCP Servers{Colors.RESET}
     Copy the configuration above to your Claude settings file

  2. {Colors.YELLOW}Install VSCode Extensions{Colors.RESET}
     Open VSCode, press Ctrl+Shift+P, and run:
     "Extensions: Show Recommended Extensions"

  3. {Colors.YELLOW}Verify Build{Colors.RESET}
     python build/build.py

  4. {Colors.YELLOW}Run Tests{Colors.RESET}
     python tests/run_tests.py

  5. {Colors.YELLOW}Run Benchmarks{Colors.RESET}
     python benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py

{Colors.BOLD}Quick Commands:{Colors.RESET}

  Build:      python build/build.py
  Test:       python tests/run_tests.py
  Benchmark:  python benchmarks/python-with-interpreter-overhead/run_all_benchmarks.py
  Lint:       ruff check .
  Format:     ruff format .
  Type check: mypy .

{Colors.BOLD}Documentation:{Colors.RESET}

  See docs/DEVELOPMENT_TOOLING.md for detailed tool documentation
""")


def main() -> int:
    """Main setup routine."""
    print(f"""
{Colors.BOLD}{Colors.BLUE}╔══════════════════════════════════════════════════════════╗
║     Ternary Engine Development Environment Setup          ║
╚══════════════════════════════════════════════════════════╝{Colors.RESET}
""")

    # Parse arguments
    full_install = "--full" in sys.argv
    check_only = "--check" in sys.argv
    skip_mcp = "--skip-mcp" in sys.argv

    # Check prerequisites
    if not check_prerequisites():
        print(f"\n{Colors.RED}Please install missing prerequisites and try again.{Colors.RESET}")
        print("\nInstallation guides:")
        print("  Node.js: https://nodejs.org/")
        print("  Python:  https://python.org/")
        print("  Git:     https://git-scm.com/")
        print("  MSVC:    Install 'Desktop development with C++' from Visual Studio Installer")
        return 1

    if check_only:
        print(f"\n{Colors.GREEN}All prerequisites are installed!{Colors.RESET}")
        return 0

    # Install tools
    install_python_tools(full=full_install)

    if not skip_mcp:
        install_mcp_servers()

    # Build and test
    build_modules()
    run_tests()

    # Generate IDE support
    generate_compile_commands()

    # Print configuration and next steps
    if not skip_mcp:
        print_mcp_config()

    print_next_steps()

    return 0


if __name__ == "__main__":
    sys.exit(main())
