import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                names.append(f"{module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
    return names


def test_story_scripts_import_the_in_repo_compressor():
    for name in ("encode_story.py", "decode_story.py"):
        path = ROOT / "core" / name
        text = path.read_text(encoding="utf-8")
        imported = _imported_names(path)
        assert "core.llmzip_utils.LLMzip" in imported
        assert "llmzip_qwen" not in text
        assert "/home/" not in text
