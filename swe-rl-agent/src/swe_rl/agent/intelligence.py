"""Bounded Python AST graph and evidence-based test recommendations."""

import ast
from pathlib import Path
from typing import Any


def analyze_repository(root: Path, max_files: int = 1000) -> dict[str, Any]:
    root = root.resolve()
    files: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if len(files) >= max_files:
            break
        if path.is_symlink() or any(
            part in {".git", ".venv", "venv", "node_modules"}
            for part in path.relative_to(root).parts
        ):
            continue
        relative = str(path.relative_to(root))
        try:
            if path.stat().st_size > 500_000:
                skipped.append(relative)
                continue
            tree = ast.parse(path.read_text(errors="replace"))
        except (SyntaxError, OSError):
            skipped.append(relative)
            continue
        symbols = []
        imports = []
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(
                    {"name": node.name, "kind": type(node).__name__, "line": node.lineno}
                )
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        files.append(
            {
                "path": relative,
                "symbols": symbols,
                "imports": imports,
                "calls": sorted(calls),
                "is_test": path.name.startswith("test_") or path.name.endswith("_test.py"),
            }
        )
    return {
        "files": files,
        "skipped": skipped,
        "limit": max_files,
        "language": "python",
        "analysis": "static AST references, not runtime call resolution",
    }


def recommend_tests(graph: dict[str, Any], changed_files: list[str]) -> list[dict[str, Any]]:
    changed = {str(Path(p).with_suffix("")).replace("/", ".") for p in changed_files}
    stems = {Path(p).stem for p in changed_files}
    symbols = {
        s["name"] for f in graph["files"] if f["path"] in changed_files for s in f["symbols"]
    }
    recommendations = []
    for f in graph["files"]:
        if not f["is_test"]:
            continue
        evidence = []
        if any(stem in Path(f["path"]).stem for stem in stems):
            evidence.append("filename matches changed module")
        if any(
            any(module.endswith(i) or i.endswith(module) for module in changed)
            for i in f["imports"]
            if i
        ):
            evidence.append("imports changed module")
        matches = symbols.intersection(f["calls"])
        if matches:
            evidence.append("references changed symbols: " + ", ".join(sorted(matches)))
        if evidence:
            recommendations.append(
                {"test_id": f["path"], "score": len(evidence), "evidence": evidence}
            )
    return sorted(recommendations, key=lambda r: (-r["score"], r["test_id"]))


def semantic_code_search(sources: dict[str, str], query: str, *, limit: int = 10) -> dict[str, Any]:
    """Rank AST definitions by conceptual term expansion and symbol/docstring relevance.

    Uses deterministic text scoring, without embeddings or inferred runtime behavior.
    Source processing is bounded to 1,000 files and two million characters.
    """
    import re

    if not query.strip() or len(query) > 2000 or not 1 <= limit <= 50:
        raise ValueError("query must contain 1..2000 characters; limit must be 1..50")

    def terms(text: str) -> set[str]:
        split = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        return {word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z0-9]*", split)}

    concepts = [
        {"serialize", "serialization", "encode", "encoding", "dump", "json"},
        {"deserialize", "decode", "decoding", "load", "parse", "parser", "parsing"},
        {"path", "url", "uri", "route", "routing", "resolve", "normalize"},
        {"cache", "cached", "memoize", "memoization"},
        {"auth", "authenticate", "authentication", "credential", "credentials", "login"},
        {"validate", "validation", "check", "verify", "verification"},
        {"error", "exception", "failure", "failed", "raise"},
    ]
    original = terms(query)
    expanded = set(original)
    for concept in concepts:
        if original.intersection(concept):
            expanded.update(concept)
    matches = []
    skipped = []
    budget = 2_000_000
    processed = 0
    for path, source in sorted(sources.items())[:1000]:
        if len(source) > 100_000 or len(source) > budget:
            skipped.append(path)
            continue
        budget -= len(source)
        processed += 1
        try:
            tree = ast.parse(source)
        except SyntaxError:
            skipped.append(path)
            continue
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            docstring = ast.get_docstring(node) or ""
            excerpt = "\n".join(
                lines[node.lineno - 1 : min(node.end_lineno or node.lineno, node.lineno + 30)]
            )[:1600]
            channels = {
                "symbol": terms(node.name),
                "docstring": terms(docstring),
                "source": terms(excerpt),
                "path": terms(path),
            }
            direct = {
                name: sorted(original.intersection(words)) for name, words in channels.items()
            }
            conceptual = {
                name: sorted((expanded - original).intersection(words))
                for name, words in channels.items()
            }
            weights = {"symbol": 5, "docstring": 3, "source": 1, "path": 1}
            score = sum(
                weights[name] * (len(direct[name]) + 0.25 * len(conceptual[name]))
                for name in channels
            )
            if score:
                matches.append(
                    {
                        "path": path,
                        "symbol": node.name,
                        "kind": type(node).__name__,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "score": score,
                        "docstring": docstring[:1000],
                        "snippet": excerpt,
                        "evidence": {"query_terms": direct, "related_terms": conceptual},
                    }
                )
    matches.sort(key=lambda entry: (-entry["score"], entry["path"], entry["line"]))
    return {
        "matches": matches[:limit],
        "files_processed": processed,
        "skipped": skipped,
        "method": "Python AST symbol/docstring/source relevance with explicit conceptual term expansion; no embeddings",
        "expanded_terms": sorted(expanded),
        "limits": {"files": 1000, "source_chars": 2_000_000, "matches": limit},
    }
