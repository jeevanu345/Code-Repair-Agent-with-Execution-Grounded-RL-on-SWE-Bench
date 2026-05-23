# CLAUDE.md (workspace-local)

This file is loaded by Claude Code when operating inside the `swe-rl-agent/` workspace.
The authoritative project-level CLAUDE.md sits at the parent directory.

Key rules to keep in mind here:

- Reward must come from real test execution inside Docker. Never mock the sandbox.
- Phases are sequential — finish Phase N's smoke test before starting N+1.
- All long-running entrypoints call `swe_rl.observability.init(component=...)`.
- All sandbox code paths use `try/finally` to guarantee container teardown.
- Tool inputs are pydantic-validated; tool outputs are structured `ToolResult`.

See [../CLAUDE.md](../CLAUDE.md) for the full operating contract.
