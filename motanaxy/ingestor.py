"""MOTANAXY Data Ingestor — Token-efficient multi-format data pipeline.

Converts any file format to clean, deduplicated training text using MarkItDown.
Applies Ponytail principles: minimal processing, maximum reuse, zero bloat.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("motanaxy.ingestor")


# ---------------------------------------------------------------------------
# Ponytail Rule: Use standard library first. Only import MarkItDown when
# the file type actually needs it (non-plaintext).
# ---------------------------------------------------------------------------

PLAINTEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".c", ".cpp", ".h",
    ".hpp", ".java", ".kt", ".scala", ".rb", ".php", ".sh", ".bash", ".zsh",
    ".ps1", ".bat", ".cmd", ".sql", ".r", ".m", ".swift", ".dart", ".lua",
    ".pl", ".ex", ".exs", ".hs", ".ml", ".clj", ".v", ".sv", ".vhd",
    ".txt", ".md", ".rst", ".csv", ".tsv", ".log", ".cfg", ".ini", ".toml",
    ".yaml", ".yml", ".json", ".xml", ".html", ".css", ".scss", ".sass",
    ".less", ".vue", ".svelte", ".astro", ".tf", ".hcl",
    ".dockerfile", ".makefile", ".cmake", ".gradle", ".sbt",
    ".gitignore", ".env", ".editorconfig",
}

MARKITDOWN_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".rtf", ".odt", ".ods", ".odp", ".epub",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
    ".wav", ".mp3", ".flac", ".ogg", ".m4a",
    ".zip", ".tar", ".gz",
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".class", ".o", ".obj", ".exe", ".dll", ".so",
    ".dylib", ".whl", ".egg", ".lock", ".bin", ".dat", ".db",
    ".sqlite", ".sqlite3", ".ico", ".svg", ".woff", ".woff2",
    ".ttf", ".eot", ".map", ".min.js", ".min.css",
}

SKIP_DIRS = {
    "__pycache__", ".git", ".svn", ".hg", ".venv", "venv", "env",
    "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "dist", "build", ".egg-info", ".eggs",
    "target", "bin", "obj", ".idea", ".vscode", ".vs",
    "runs",  # Don't ingest our own model outputs
}

# Max file size to ingest (Ponytail: don't burn resources on giant files)
MAX_FILE_BYTES = 512 * 1024  # 512 KB


class DataIngestor:
    """Efficiently reads files from disk with minimal token waste.

    Strategy:
    1. Plaintext/code files → read directly (zero overhead)
    2. Documents/media → MarkItDown converts to markdown
    3. Deduplication by content hash (no repeated data in training)
    4. Cleaning pipeline strips noise (blank lines, comments-only, etc.)
    """

    def __init__(self, use_markitdown: bool = True):
        self._md = None
        self._use_markitdown = use_markitdown
        self._seen_hashes: set[str] = set()
        self.stats = {
            "files_scanned": 0,
            "files_ingested": 0,
            "files_skipped": 0,
            "files_duplicate": 0,
            "bytes_raw": 0,
            "bytes_clean": 0,
        }

    @property
    def md(self):
        """Lazy-load MarkItDown only when needed (Ponytail: defer work)."""
        if self._md is None and self._use_markitdown:
            try:
                from markitdown import MarkItDown
                self._md = MarkItDown()
                logger.info("MarkItDown loaded successfully")
            except ImportError:
                logger.warning(
                    "markitdown not installed. "
                    "Install with: pip install 'markitdown[all]'"
                )
                self._use_markitdown = False
        return self._md

    # ----- Core reading methods -----

    def read_plaintext(self, path: Path) -> Optional[str]:
        """Read a text file with minimal overhead."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug("Skip %s: %s", path, exc)
            return None

    def read_with_markitdown(self, path: Path) -> Optional[str]:
        """Convert a non-text file to markdown via MarkItDown."""
        if not self.md:
            return None
        try:
            result = self.md.convert(str(path))
            return result.text_content
        except Exception as exc:
            logger.debug("MarkItDown failed on %s: %s", path, exc)
            return None

    # ----- Cleaning pipeline (Ponytail: strip noise, keep signal) -----

    @staticmethod
    def clean(text: str) -> str:
        """Clean text for training: remove excessive whitespace and noise."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse 3+ blank lines to 2 (keep document separation)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        # Strip trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        # Strip leading/trailing whitespace of the whole document
        text = text.strip()
        return text

    @staticmethod
    def is_meaningful(text: str, min_bytes: int = 64) -> bool:
        """Check if text has enough signal for training."""
        if len(text.encode("utf-8")) < min_bytes:
            return False
        # Skip files that are almost entirely comments or blank
        lines = [l for l in text.split("\n") if l.strip()]
        code_lines = [l for l in lines if not l.strip().startswith("#")]
        if len(lines) > 3 and len(code_lines) < len(lines) * 0.15:
            return False
        return True

    # ----- Deduplication -----

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _is_duplicate(self, text: str) -> bool:
        h = self._content_hash(text)
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    # ----- Main ingestion -----

    def ingest_file(self, path: Path) -> Optional[str]:
        """Ingest a single file and return cleaned text, or None."""
        self.stats["files_scanned"] += 1

        # Size guard
        try:
            size = path.stat().st_size
        except OSError:
            self.stats["files_skipped"] += 1
            return None

        if size > MAX_FILE_BYTES or size == 0:
            self.stats["files_skipped"] += 1
            return None

        ext = path.suffix.lower()
        name = path.name.lower()

        # Skip binary/generated files
        if ext in SKIP_EXTENSIONS:
            self.stats["files_skipped"] += 1
            return None

        # Read content
        if ext in PLAINTEXT_EXTENSIONS or name in {
            "dockerfile", "makefile", "cmakelists.txt", "rakefile",
            "gemfile", "procfile", "vagrantfile",
        }:
            raw = self.read_plaintext(path)
        elif ext in MARKITDOWN_EXTENSIONS:
            raw = self.read_with_markitdown(path)
        else:
            # Unknown extension: try plaintext first
            raw = self.read_plaintext(path)

        if raw is None:
            self.stats["files_skipped"] += 1
            return None

        self.stats["bytes_raw"] += len(raw.encode("utf-8"))

        # Clean
        cleaned = self.clean(raw)

        # Quality check
        if not self.is_meaningful(cleaned):
            self.stats["files_skipped"] += 1
            return None

        # Deduplicate
        if self._is_duplicate(cleaned):
            self.stats["files_duplicate"] += 1
            return None

        self.stats["files_ingested"] += 1
        self.stats["bytes_clean"] += len(cleaned.encode("utf-8"))
        return cleaned

    def ingest_directory(self, root: Path, recursive: bool = True) -> list[str]:
        """Walk a directory tree, ingest all eligible files."""
        documents: list[str] = []
        root = root.resolve()

        if not root.is_dir():
            logger.error("%s is not a directory", root)
            return documents

        walker = root.rglob("*") if recursive else root.glob("*")
        for path in sorted(walker):
            # Skip excluded directories
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            text = self.ingest_file(path)
            if text:
                documents.append(text)

        return documents

    def ingest_paths(self, paths: list[str]) -> list[str]:
        """Ingest a list of files and/or directories."""
        documents: list[str] = []
        for p in paths:
            path = Path(p).resolve()
            if path.is_dir():
                documents.extend(self.ingest_directory(path))
            elif path.is_file():
                text = self.ingest_file(path)
                if text:
                    documents.append(text)
            else:
                logger.warning("Path not found: %s", path)
        return documents


def build_corpus(documents: list[str], separator: str = "\n\n\n") -> str:
    """Join documents into a single training corpus.

    Uses triple-newline separator so the model learns document boundaries.
    This matches code_ai.py's expected format.
    """
    return separator.join(documents)


def corpus_stats(corpus: str) -> dict:
    """Return summary statistics of a corpus."""
    raw_bytes = corpus.encode("utf-8")
    docs = [d for d in corpus.split("\n\n\n") if d.strip()]
    return {
        "documents": len(docs),
        "total_bytes": len(raw_bytes),
        "total_chars": len(corpus),
        "unique_bytes": len(set(raw_bytes)),
        "avg_doc_bytes": len(raw_bytes) // max(len(docs), 1),
    }
