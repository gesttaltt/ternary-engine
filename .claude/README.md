# Claude Code Configuration for Ternary Engine

This directory contains project-specific configuration, commands, and context for Claude Code sessions.

## Purpose

The `.claude/` directory provides Claude Code with:
- **Project-specific standards** (coding conventions, architecture principles)
- **Slash commands** for common workflows (build, test, benchmark)
- **Context documents** for quick reference during development
- **Templates** for consistent code generation (future)

## Structure

```
.claude/
├── CLAUDE.md              # Main project configuration
├── README.md              # This file
├── commands/              # Slash commands
│   ├── build.md               # Build standard module
│   ├── build-dense243.md      # Build Dense243 module
│   ├── test.md                # Run test suite
│   ├── benchmark.md           # Run performance benchmarks
│   ├── tritnet.md             # TritNet workflow
│   ├── clean.md               # Clean build artifacts
│   └── timestamp.md           # Create IP timestamp
├── context/               # Context documents for quick reference
│   ├── codebase-overview.md   # Project structure and navigation
│   ├── architecture.md        # System architecture and design
│   └── tritnet-context.md     # TritNet implementation guide
└── templates/             # Code templates (future use)
```

## Main Configuration (CLAUDE.md)

**File:** `.claude/CLAUDE.md`

Defines project-level standards for:
- **Code quality:** YAGNI, phase coherence, performance requirements
- **Architecture:** Kernel vs engine separation, design principles
- **Testing:** Coverage requirements, test categories
- **Performance:** Optimization hierarchy, benchmarking standards
- **TritNet:** Development phases, training guidelines
- **Build system:** Platform requirements, compiler flags
- **Documentation:** Standards and format requirements

**Key principles:**
- YAGNI (You Aren't Gonna Need It) - No speculative code
- Phase coherence - Only add complexity if >10% performance gain
- Benchmark everything - All optimizations validated
- Windows x64 only - No production claims until platform validated

## Slash Commands

**Usage:** Type `/command` in Claude Code to execute

### Available Commands

**Build and development:**
- `/build` - Build standard ternary_simd_engine module
- `/build-dense243` - Build Dense243 high-density encoding module
- `/clean` - Clean all build artifacts and temporary files

**Testing and benchmarking:**
- `/test` - Run comprehensive test suite (65 tests)
- `/benchmark` - Run performance benchmarks (core + competitive)

**TritNet workflow:**
- `/tritnet` - TritNet neural network training and workflows

**IP protection:**
- `/timestamp` - Create OpenTimestamps blockchain timestamp

### Creating New Commands

1. Create new file in `.claude/commands/` (e.g., `mycommand.md`)
2. Write command description and bash code blocks
3. Use the command with `/mycommand`

Example:
```markdown
# .claude/commands/mycommand.md
Brief description of what this command does.

```bash
python scripts/my_script.py --option value
```

Additional notes or usage instructions.
```

## Context Documents

**Purpose:** Provide quick reference without searching codebase

### Codebase Overview (`context/codebase-overview.md`)
- Directory structure and file organization
- Key files by purpose
- Technology stack
- Quick commands
- Performance highlights

**Use when:** Navigating codebase, understanding project structure

### Architecture Guide (`context/architecture.md`)
- System architecture layers (0-5)
- Kernel vs engine separation
- Design principles (YAGNI, phase coherence, etc.)
- Data flow and performance characteristics
- TritNet architecture overview

**Use when:** Making architectural decisions, understanding design

### TritNet Context (`context/tritnet-context.md`)
- TritNet vision and innovation
- Model architectures (TritNetUnary, TritNetBinary)
- Training process and phases
- Implementation details
- Performance expectations
- Critical success factors

**Use when:** Working on TritNet, understanding neural network approach

## Templates (Future)

**Directory:** `.claude/templates/`

Will contain:
- **C++ header template** - Standard header file format
- **Python script template** - Script format with copyright
- **Test file template** - Test file structure
- **Documentation template** - Consistent doc format

**Status:** Not yet implemented (to be added as needed)

## Integration with User-Level Config

**User-level config:** `~/.claude/CLAUDE.md`
**Project-level config:** `./.claude/CLAUDE.md` (this directory)

**Inheritance:**
- User-level defines global preferences (communication style, security)
- Project-level extends with Ternary Engine-specific standards
- Project-level NEVER contradicts user-level security policies

**Example:**
- User config: "No emojis, professional B2B style"
- Project config: "YAGNI principle, >10% performance threshold"
- Result: Both apply during Ternary Engine sessions

## Using This Configuration

### In Claude Code Sessions

**Automatic loading:**
- Claude Code automatically reads `.claude/CLAUDE.md` when starting
- Context documents available for quick reference
- Slash commands registered and ready to use

**Explicit reference:**
- "Following the YAGNI principle defined in .claude/CLAUDE.md..."
- "Using the architecture defined in .claude/context/architecture.md..."
- "As specified in project standards..."

### Updating Configuration

**When to update CLAUDE.md:**
- New architectural patterns emerge
- Performance thresholds change
- New development phases added
- Build system changes
- New critical paths identified

**When to add slash commands:**
- Repetitive workflows identified
- Common multi-step operations
- Frequently needed actions

**When to update context documents:**
- Major refactoring completed
- New features added
- Architecture evolved
- TritNet phases progress

## Best Practices

### For Contributors

1. **Read CLAUDE.md first** - Understand project standards before coding
2. **Use slash commands** - Consistent workflows across team
3. **Reference context docs** - Quick lookups without searching
4. **Update when evolving** - Keep docs in sync with codebase

### For Claude Code Sessions

1. **Cite standards** - Reference .claude/CLAUDE.md when making decisions
2. **Use templates** - Consistent code generation (when implemented)
3. **Follow workflows** - Use slash commands for standard operations
4. **Maintain context** - Update docs when architecture changes

### For Maintainers

1. **Keep CLAUDE.md current** - Update when standards evolve
2. **Add commands as needed** - Don't over-engineer, add when useful
3. **Prune stale content** - Remove outdated information
4. **Version control** - Track changes to configuration

## Changelog

| Date       | Version | Description                              |
|:-----------|:--------|:-----------------------------------------|
| 2025-11-23 | v1.0.0  | Initial .claude configuration created    |

## Future Enhancements

**Planned additions:**
- Code templates for consistency
- Additional slash commands (PGO build, fusion benchmarks)
- Visual architecture diagrams
- Performance regression tracking
- Automated documentation generation

**Long-term vision:**
- Integration with CI/CD workflows
- Automated code review checks against standards
- Performance monitoring dashboards
- TritNet training automation

## Support

**Questions about this configuration?**
- Review `.claude/CLAUDE.md` for detailed standards
- Check `.claude/context/` for quick reference guides
- See main project documentation in `docs/`

**Contributing to configuration:**
- Propose changes via pull requests
- Discuss major changes with team first
- Keep standards practical and measurable

---

**Remember:** This configuration exists to make development faster and more consistent. If something isn't working, update it. If a standard is unclear, clarify it. The goal is to help, not hinder.

---

**Version:** 1.0.0 · **Created:** 2025-11-23 · **Project:** Ternary Engine
