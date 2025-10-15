# GitHub Actions Configuration Guide

This document describes how to configure the CI/CD pipeline using GitHub repository secrets and variables.

## Overview

The workflows support configuration through:
- **Environment Variables** - Defined at the workflow level with sensible defaults
- **Repository Variables** - Override workflow-level settings (Settings → Secrets and variables → Actions → Variables)
- **Repository Secrets** - Override dependency versions and enable verbose modes (Settings → Secrets and variables → Actions → Secrets)

## Quick Start

The workflows work out-of-the-box with default settings. Only configure secrets/variables if you need to:
- Pin specific dependency versions
- Adjust timeout values
- Enable verbose build/test output
- Change artifact retention

## Repository Variables (Optional)

These override workflow defaults. Navigate to: **Settings → Secrets and variables → Actions → Variables**

| Variable Name | Type | Default | Description |
|---------------|------|---------|-------------|
| `TEST_TIMEOUT_MINUTES` | Number | `10` | Timeout for test jobs (CI workflow) |
| `BENCHMARK_TIMEOUT_MINUTES` | Number | `30` | Timeout for benchmark jobs |
| `PGO_TIMEOUT_MINUTES` | Number | `45` | Timeout for PGO benchmark workflow |
| `ARTIFACT_RETENTION_DAYS` | Number | `7` | How long to keep test artifacts |
| `BENCHMARK_RETENTION_DAYS` | Number | `30` | How long to keep benchmark results |

### Example: Increase Test Timeout

1. Go to **Settings → Secrets and variables → Actions → Variables**
2. Click **New repository variable**
3. Name: `TEST_TIMEOUT_MINUTES`
4. Value: `15`
5. Click **Add variable**

## Repository Secrets (Optional)

These override dependency versions and control verbosity. Navigate to: **Settings → Secrets and variables → Actions → Secrets**

### Dependency Version Control

| Secret Name | Default | Description |
|-------------|---------|-------------|
| `PYBIND11_VERSION` | `latest` | Pin pybind11 version (e.g., `2.11.1`) |
| `NUMPY_VERSION` | `latest` | Pin numpy version (e.g., `1.24.0`) |

**When to use:**
- Lock to specific tested versions for reproducibility
- Work around breaking changes in new releases
- Test compatibility with specific versions

### Build & Test Configuration

| Secret Name | Default | Description |
|-------------|---------|-------------|
| `BUILD_VERBOSE` | `0` | Enable verbose build output (`1` = verbose) |
| `TEST_VERBOSE` | `0` | Enable verbose test output (`1` = verbose) |
| `BENCHMARK_ITERATIONS` | `1000` | Number of benchmark iterations |

**When to use:**
- Debugging build failures (set `BUILD_VERBOSE=1`)
- Investigating test issues (set `TEST_VERBOSE=1`)
- Faster benchmarks during development (lower `BENCHMARK_ITERATIONS`)

### Example: Pin NumPy Version

1. Go to **Settings → Secrets and variables → Actions → Secrets**
2. Click **New repository secret**
3. Name: `NUMPY_VERSION`
4. Value: `1.24.3`
5. Click **Add secret**

### Example: Enable Verbose Build Output

1. Go to **Settings → Secrets and variables → Actions → Secrets**
2. Click **New repository secret**
3. Name: `BUILD_VERBOSE`
4. Value: `1`
5. Click **Add secret**

## Workflow-Specific Environment Variables

These are defined in each workflow file and can be viewed/modified there:

### CI Workflow (`ci.yml`)

```yaml
env:
  PYTHON_VERSION_DEFAULT: '3.11'
  MSVC_ARCH: 'x64'
  COMPILER_FLAGS_WINDOWS: '/O2 /GL /arch:AVX2 /openmp /std:c++17'
  COMPILER_FLAGS_LINUX: '-O3 -march=haswell -mavx2 -fopenmp -std=c++17'
```

### Benchmarks Workflow (`benchmarks.yml`)

```yaml
env:
  PYTHON_VERSION: '3.11'
  BENCHMARK_MODE: 'quick'  # or 'full'
  MSVC_ARCH: 'x64'
```

### PGO Benchmark Workflow (`pgo_benchmark.yml`)

```yaml
env:
  PYTHON_VERSION: '3.11'
  MSVC_ARCH: 'x64'
  PGO_MODE: 'full'
```

## Manual Workflow Triggers

### Benchmarks Workflow

Supports manual dispatch with inputs:

1. Go to **Actions → Performance Benchmarks → Run workflow**
2. Configure:
   - **Branch**: Select branch to run on
   - **Compare against baseline**: `true` or `false`
   - **Benchmark mode**: `quick` or `full`
3. Click **Run workflow**

### PGO Benchmark Workflow

Supports manual dispatch with inputs:

1. Go to **Actions → Run PGO Benchmark → Run workflow**
2. Configure:
   - **Branch**: Select branch to run on
   - **Benchmark mode**: `quick` or `full`
3. Click **Run workflow**

## Configuration Examples

### Development Environment

Fast feedback, verbose output:

**Repository Secrets:**
- `BUILD_VERBOSE`: `1`
- `TEST_VERBOSE`: `1`
- `BENCHMARK_ITERATIONS`: `100`

**Repository Variables:**
- `TEST_TIMEOUT_MINUTES`: `5`
- `BENCHMARK_TIMEOUT_MINUTES`: `10`
- `ARTIFACT_RETENTION_DAYS`: `3`

### Production Environment

Locked versions, thorough testing:

**Repository Secrets:**
- `PYBIND11_VERSION`: `2.11.1`
- `NUMPY_VERSION`: `1.24.3`
- `BENCHMARK_ITERATIONS`: `10000`

**Repository Variables:**
- `TEST_TIMEOUT_MINUTES`: `15`
- `BENCHMARK_TIMEOUT_MINUTES`: `60`
- `ARTIFACT_RETENTION_DAYS`: `30`

### CI Cost Optimization

Minimize runner time and storage:

**Repository Variables:**
- `TEST_TIMEOUT_MINUTES`: `8`
- `BENCHMARK_TIMEOUT_MINUTES`: `20`
- `ARTIFACT_RETENTION_DAYS`: `3`
- `BENCHMARK_RETENTION_DAYS`: `7`

## Troubleshooting

### Build Failures

1. Enable verbose output:
   - Set `BUILD_VERBOSE` secret to `1`
2. Check compiler environment:
   - Verify MSVC setup step succeeded (Windows)
   - Verify GCC installation (Linux)
3. Review build artifacts:
   - Download artifacts from failed workflow run
   - Check `build/artifacts/` directory

### Test Failures

1. Enable verbose test output:
   - Set `TEST_VERBOSE` secret to `1`
2. Download test artifacts:
   - Go to workflow run → Artifacts
   - Download `test-results-*` artifact
3. Check timeout settings:
   - Increase `TEST_TIMEOUT_MINUTES` if tests are being killed

### Benchmark Issues

1. Verify module built successfully:
   - Check "Verify module was built" step in workflow
   - Look for `.pyd` (Windows) or `.so` (Linux) file
2. Adjust benchmark mode:
   - Use `quick` mode for faster runs
   - Use `full` mode for comprehensive results
3. Check iterations:
   - Lower `BENCHMARK_ITERATIONS` for faster debugging
   - Default `1000` is reasonable for CI

## Security Notes

- **Do NOT** store actual credentials as secrets (this project doesn't require any)
- Repository variables are visible to all users with read access
- Repository secrets are encrypted and only visible to maintainers
- Workflow logs may expose environment variable values (but not secrets)

## Reference

- [GitHub Actions: Using secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions)
- [GitHub Actions: Using variables](https://docs.github.com/en/actions/learn-github-actions/variables)
- [GitHub Actions: Workflow syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

---

**Last Updated**: 2025-10-15
**Workflows**: `ci.yml`, `benchmarks.yml`, `pgo_benchmark.yml`
