"""Sandbox repository intelligence with bounded source collection."""

import json
import shlex
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, Field

from swe_rl.agent.intelligence import analyze_repository, recommend_tests, semantic_code_search
from swe_rl.agent.tools.base import Tool, ToolResult, truncate


class IntelligenceArgs(BaseModel):
    changed_files: list[str] = Field(default_factory=list)
    max_files: int = Field(200, ge=1, le=1000)


class RepositoryGraphTool(Tool):
    name = "repository_graph"
    description = "Inspect Python symbols, imports, static calls and recommend related tests for changed files."
    Args = IntelligenceArgs

    def _run(self, args: IntelligenceArgs) -> ToolResult:
        result = _collect_sources(self.ctx, args.max_files)
        if result.exit_code:
            return ToolResult(False, result.stdout, {}, result.duration_s)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, source in json.loads(result.stdout).items():
                destination = (root / name).resolve()
                if destination.is_relative_to(root.resolve()):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(source)
            graph = analyze_repository(root, args.max_files)
        data = {"graph": graph, "recommended_tests": recommend_tests(graph, args.changed_files)}
        output, truncated = truncate(json.dumps(data))
        return ToolResult(True, output, data, result.duration_s, truncated)


class HistoryArgs(BaseModel):
    path: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=30)


class GitHistoryTool(Tool):
    name = "git_history"
    description = "Retrieve recent commits and patches affecting a relative repository path."
    Args = HistoryArgs

    def _run(self, args: HistoryArgs) -> ToolResult:
        path = Path(args.path)
        if path.is_absolute() or ".." in path.parts:
            return ToolResult(False, "path must remain inside repository", {}, 0.0)
        command = f"cd {shlex.quote(self.ctx.repo_dir)} && git log -n {args.limit} --format='%h %s' -p -- {shlex.quote(args.path)}"
        result = self.ctx.runner.exec(self.ctx.handle, command, check_safety=False)
        output, truncated = truncate(result.stdout)
        return ToolResult(
            result.exit_code == 0, output, {"path": args.path}, result.duration_s, truncated
        )


class SymbolEditArgs(BaseModel):
    path: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=1)
    replacement: str = Field(..., min_length=1)


class SymbolEditTool(Tool):
    name = "symbol_edit"
    description = "Replace one uniquely named Python function or class; validate resulting AST before writing. Relative Python paths only."
    Args = SymbolEditArgs

    def _run(self, args: SymbolEditArgs) -> ToolResult:
        import ast

        path = Path(args.path)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            return ToolResult(False, "requires relative Python path inside repository", {}, 0)
        read = self.ctx.runner.exec(
            self.ctx.handle,
            "cd " + shlex.quote(self.ctx.repo_dir) + " && cat -- " + shlex.quote(args.path),
            check_safety=False,
        )
        if read.exit_code:
            return ToolResult(False, read.stdout, {}, read.duration_s)
        try:
            tree = ast.parse(read.stdout)
            nodes = [
                n
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.name == args.symbol
            ]
            if len(nodes) != 1:
                return ToolResult(
                    False, "symbol must match exactly one definition", {"matches": len(nodes)}, 0
                )
            node = nodes[0]
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            lines = read.stdout.splitlines(keepends=True)
            indent = lines[node.lineno - 1][: node.col_offset]
            replacement = "".join(
                indent + line if line.strip() else line
                for line in (args.replacement.rstrip() + "\n").splitlines(keepends=True)
            )
            source = "".join(lines[: start - 1]) + replacement + "".join(lines[node.end_lineno :])
            ast.parse(source)
        except SyntaxError as exc:
            return ToolResult(False, f"AST validation failed: {exc}", {}, 0)
        self.ctx.runner.write_file(self.ctx.handle, f"{self.ctx.repo_dir}/{args.path}", source)
        return ToolResult(
            True,
            "Symbol replaced; Python syntax validated.",
            {
                "path": args.path,
                "symbol": args.symbol,
                "start_line": start,
                "end_line": node.end_lineno,
            },
            0,
        )


def _collect_sources(ctx, max_files):
    script = (
        "import pathlib,json; r=pathlib.Path('.'); fs=[p for p in r.rglob('*.py') if not p.is_symlink() and not any(x in {'.git','.venv','venv','node_modules'} for x in p.parts)]; out={}; budget=2000000\nfor p in sorted(fs)[:"
        + str(max_files)
        + "]:\n try:\n  size=p.stat().st_size\n  if size<=100000 and size<=budget:\n   out[str(p)]=p.read_text(errors='replace'); budget-=size\n except OSError: pass\nprint(json.dumps(out))"
    )
    return ctx.runner.exec(
        ctx.handle,
        "cd " + shlex.quote(ctx.repo_dir) + " && python -c " + shlex.quote(script),
        check_safety=False,
    )


class SemanticSearchArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(10, ge=1, le=50)
    max_files: int = Field(200, ge=1, le=1000)


class SemanticCodeSearchTool(Tool):
    name = "semantic_code_search"
    description = "Search Python definitions by symbol/docstring/source relevance and related concepts. Bounded AST/text scoring with explicit evidence; no embeddings."
    Args = SemanticSearchArgs

    def _run(self, args: SemanticSearchArgs) -> ToolResult:
        result = _collect_sources(self.ctx, args.max_files)
        if result.exit_code:
            return ToolResult(False, result.stdout, {}, result.duration_s)
        try:
            data = semantic_code_search(json.loads(result.stdout), args.query, limit=args.limit)
        except (ValueError, TypeError) as exc:
            return ToolResult(False, str(exc), {}, result.duration_s)
        output, truncated = truncate(json.dumps(data))
        return ToolResult(True, output, data, result.duration_s, truncated)
