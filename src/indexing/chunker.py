"""
Structural chunking over the d2l-en corpus. -> component docs/indexing.md, section 2

Pipeline (mirrors indexing.md, section 5's pseudocode, steps 1-2; embedding/FAISS are
embedder.py / build_index.py's job, not this module's):

    walk_repo -> preclean (2.0) -> parse heading hierarchy -> split into prose/code segments,
    merging short ones -> enforce max_tokens -> attach metadata -> Chunk objects

Entry point: `chunk_repo(root_dir)` returns the full list of `Chunk`s for every `chapter_*/`
file in the corpus, ready to hand to `embedder.py`.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# -- Tunables (indexing.md, section 2) ---------------------------------------------------

KEEP_FRAMEWORK = "pytorch"     # section 2.0 -- the only framework variant we keep
MAX_TOKENS = 500               # section 2, "Chunk boundary rule"
MERGE_SHORT_HEADING_TOKENS = 40  # section 2, "very short" heading-content threshold
SHORT_CODE_LINES = 5             # section 2, short-code-cell merge threshold
EXCLUDED_HEADINGS = {"exercises"}  # section 2, content_type "exercises" -- excluded for v1


def _approx_token_count(text: str) -> int:
    """Whitespace word count as a token-count approximation -- no tokenizer dependency.
    Good enough for the ~500-token threshold in indexing.md; revisit if eval shows drift
    from the embedding model's actual tokenizer."""
    return len(text.split())


# -- Chunk record -------------------------------------------------------------------------

@dataclass
class Chunk:
    """Matches the metadata schema in indexing.md, section 2. `label` and `has_math` are
    the doc's optional additions beyond CLAUDE.md's required 5 fields."""

    chunk_id: str
    chapter: str
    section: str
    heading_path: list[str]
    content_type: str          # "prose" | "code"
    source_path: str
    text: str
    label: str | None = None
    has_math: bool = False

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "chapter": self.chapter,
            "section": self.section,
            "heading_path": self.heading_path,
            "content_type": self.content_type,
            "source_path": self.source_path,
            "text": self.text,
            "label": self.label,
            "has_math": self.has_math,
        }


# -- 2.0 Pre-cleaning pass ------------------------------------------------------------------

_BOILERPLATE_CELL_RE = re.compile(
    r"\A```\{[^\n`]*\}\n(?:(?!```).)*?tab\.interact_select(?:(?!```).)*?\n```\n*",
    re.DOTALL,
)
_TRAILING_DISCUSSION_RE = re.compile(
    r"\n*:begin_tab:`[\w-]+`\s*\n\[Discussions\]\([^)]*\)\s*\n:end_tab:\s*", re.DOTALL
)
_CODE_FENCE_RE = re.compile(r"```(\{[^\n`]*\})?\n(.*?)\n```", re.DOTALL)
_TAB_DIRECTIVE_RE = re.compile(r"^[ \t]*(?:%%tab|#@tab)\s+(.+?)[ \t]*$", re.MULTILINE)
_BEGIN_TAB_RE = re.compile(r":begin_tab:`([\w-]+)`\s*\n?(.*?):end_tab:", re.DOTALL)
_LABEL_RE = re.compile(r":label:`([\w-]+)`")
_DIRECTIVE_TOKEN_RE = re.compile(r":(?:eqlabel|numref|eqref|cite|citet):`[^`]*`")
_HIGHLIGHT_BOLD_PAREN_RE = re.compile(r"\(\*\*(.*?)\*\*\)", re.DOTALL)
_HIGHLIGHT_STRIKE_PAREN_RE = re.compile(r"\(~~.*?~~\)", re.DOTALL)
_HIGHLIGHT_BRACKET_RE = re.compile(r"\[\*\*(.*?)\*\*\]", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[(.*?)\]\([^)]*\)")


def _drop_boilerplate(text: str) -> str:
    """Strip the opening `%load_ext d2lbook.tab` / `tab.interact_select(...)` cell and the
    trailing per-framework `[Discussions](...)` blocks. -> indexing.md, section 2.0"""
    text = _BOILERPLATE_CELL_RE.sub("", text, count=1)
    text = _TRAILING_DISCUSSION_RE.sub("", text)
    return text


def _keep_code_cell(tab_line: str, keep_framework: str) -> bool:
    frameworks = [t.strip().lower() for t in tab_line.split(",")]
    return "all" in frameworks or keep_framework in frameworks


def _filter_code_cells(text: str, keep_framework: str) -> str:
    """Keep `%%tab all` / `%%tab ...,<keep_framework>,...` code cells, drop the rest, and
    strip the `%%tab ...` directive line itself from cells we keep. -> section 2.0"""

    def _replace(match: re.Match) -> str:
        info, body = match.group(1) or "", match.group(2)
        tab_match = _TAB_DIRECTIVE_RE.match(body)
        if tab_match is None:
            # No %%tab directive at all -- framework-agnostic cell (e.g. plain setup); keep.
            return match.group(0)
        if not _keep_code_cell(tab_match.group(1), keep_framework):
            return ""
        body = _TAB_DIRECTIVE_RE.sub("", body, count=1).strip("\n")
        return f"```{info}\n{body}\n```"

    return _CODE_FENCE_RE.sub(_replace, text)


def _filter_prose_tabs(text: str, keep_framework: str) -> str:
    """Unwrap `:begin_tab:`<keep_framework>`` blocks to their inner text; drop other
    frameworks' blocks entirely. -> section 2.0"""

    def _replace(match: re.Match) -> str:
        framework, body = match.group(1).lower(), match.group(2)
        return body.strip("\n") if framework == keep_framework else ""

    return _BEGIN_TAB_RE.sub(_replace, text)


def _strip_directives(text: str) -> str:
    """Remove `:eqlabel:`/`:numref:`/`:eqref:`/`:cite:`/`:citet:` tokens. `:label:` is
    handled separately by `_extract_labels` (it's captured as metadata, not just dropped).
    -> section 2.0"""
    text = _DIRECTIVE_TOKEN_RE.sub("", text)
    text = _LABEL_RE.sub("", text)
    return text


def _unwrap_highlights(text: str) -> str:
    """`(**x**)` -> `x`, `(~~x~~)` -> dropped, `[**x**]` -> `x`. -> section 2.0"""
    text = _HIGHLIGHT_STRIKE_PAREN_RE.sub("", text)
    text = _HIGHLIGHT_BOLD_PAREN_RE.sub(r"\1", text)
    text = _HIGHLIGHT_BRACKET_RE.sub(r"\1", text)
    return text


def _images_to_alt_text(text: str) -> str:
    """`![alt](path)` -> `alt` (the path is dead once `img/` is stripped locally). -> section 2.0"""
    return _IMAGE_RE.sub(r"\1", text)


def _extract_labels(text: str) -> dict[int, str]:
    """Map each heading line's index (by line number) to the `:label:` value that
    immediately follows it, if any -- so it can be attached to that heading's chunk(s)
    before `:label:` tokens are stripped from the body text."""
    labels: dict[int, str] = {}
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            for j in range(i + 1, min(i + 3, len(lines))):
                m = _LABEL_RE.search(lines[j])
                if m:
                    labels[i] = m.group(1)
                    break
                if lines[j].strip():
                    break  # non-blank, non-label line -- label doesn't apply here
    return labels


def preclean(raw_markdown: str, keep_framework: str = KEEP_FRAMEWORK) -> tuple[str, dict[int, str]]:
    """Full pre-cleaning pass. -> indexing.md, section 2.0. Returns (cleaned_text,
    heading_line_index -> label) since label extraction has to happen before headings are
    parsed but after boilerplate/tab-filtering (labels can appear inside dropped blocks)."""
    text = _drop_boilerplate(raw_markdown)
    text = _filter_code_cells(text, keep_framework)
    text = _filter_prose_tabs(text, keep_framework)
    labels = _extract_labels(text)
    text = _strip_directives(text)
    text = _unwrap_highlights(text)
    text = _images_to_alt_text(text)
    return text, labels


# -- Heading hierarchy parsing --------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


@dataclass
class _Leaf:
    """One heading's own immediate content (not its children's) -- the base chunking unit
    per indexing.md's "default boundary = each ## (or lowest heading level present) block"
    rule, generalized to whichever level (H1 intro / H2 / H3) actually has content."""

    heading_path: list[str]
    label: str | None
    raw_content: str


def _parse_leaves(cleaned_text: str, labels_by_line: dict[int, str]) -> list[_Leaf]:
    lines = cleaned_text.split("\n")
    heading_positions: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            heading_positions.append((i, len(m.group(1)), m.group(2).strip()))

    leaves: list[_Leaf] = []
    stack: list[str] = []  # current heading_path ancestry

    for idx, (line_idx, level, title) in enumerate(heading_positions):
        # Own content = everything up to the next heading of any level.
        end_line = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else len(lines)
        content_start = line_idx + 1
        own_content = "\n".join(lines[content_start:end_line]).strip("\n")

        stack = stack[: level - 1] + [title]
        leaves.append(_Leaf(heading_path=list(stack), label=labels_by_line.get(line_idx), raw_content=own_content))

    return leaves


# -- Prose/code segmentation within a leaf ---------------------------------------------------

@dataclass
class _Segment:
    content_type: str  # "prose" | "code"
    text: str


def _segment_leaf(raw_content: str) -> list[_Segment]:
    """Split a leaf's own content into ordered prose/code segments, merging short code
    cells into the preceding prose paragraph. -> indexing.md, section 2, "Chunk boundary
    rule" (code-block-boundary splitting + short-code merge)."""
    if not raw_content.strip():
        return []

    segments: list[_Segment] = []
    pos = 0
    for match in _CODE_FENCE_RE.finditer(raw_content):
        prose_before = raw_content[pos : match.start()].strip()
        if prose_before:
            segments.append(_Segment("prose", prose_before))

        code_text = match.group(0)
        code_lines = [ln for ln in match.group(2).split("\n") if ln.strip()]
        if len(code_lines) < SHORT_CODE_LINES and segments and segments[-1].content_type == "prose":
            # Short code cell -> merge into the immediately preceding prose paragraph.
            segments[-1] = _Segment("prose", segments[-1].text + "\n\n" + code_text)
        else:
            segments.append(_Segment("code", code_text))

        pos = match.end()

    trailing_prose = raw_content[pos:].strip()
    if trailing_prose:
        segments.append(_Segment("prose", trailing_prose))

    return segments


# -- Size enforcement + has_math ------------------------------------------------------------

_MATH_DENSE_RE = re.compile(r"\$")


def _has_math(text: str) -> bool:
    """`$$` display math, or dense enough inline `$...$` usage. -> section 2, content-type
    tagging note on math staying inline in prose chunks."""
    if "$$" in text:
        return True
    return len(_MATH_DENSE_RE.findall(text)) >= 4


def _split_prose_to_limit(text: str, max_tokens: int) -> list[str]:
    """Split an over-long prose segment at paragraph breaks, greedily filling each chunk
    up to `max_tokens`. -> section 2, "prefer splitting at paragraph breaks"."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        para_tokens = _approx_token_count(para)
        if current and current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += para_tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# -- Top-level chunking ----------------------------------------------------------------------

def chunk_markdown(
    raw_markdown: str,
    *,
    chapter: str,
    section: str,
    source_path: str,
    keep_framework: str = KEEP_FRAMEWORK,
    max_tokens: int = MAX_TOKENS,
) -> list[Chunk]:
    """Chunk a single pre-loaded markdown file's raw text. See `chunk_file`/`chunk_repo`
    for the filesystem-walking entry points."""
    cleaned, labels_by_line = preclean(raw_markdown, keep_framework)
    leaves = _parse_leaves(cleaned, labels_by_line)

    # Drop excluded headings (Exercises) before doing anything else with them.
    leaves = [leaf for leaf in leaves if (leaf.heading_path[-1].strip().lower() if leaf.heading_path else "") not in EXCLUDED_HEADINGS]

    # Build (leaf, segments) pairs, then fold "very short" heading blocks forward into the
    # next leaf (or backward if it's the last leaf in the file). -> section 2, "very short"
    # heading-content merge rule; "Summary" is exempt.
    leaf_segments: list[tuple[_Leaf, list[_Segment]]] = [(leaf, _segment_leaf(leaf.raw_content)) for leaf in leaves]
    leaf_segments = [(leaf, segs) for leaf, segs in leaf_segments if segs]

    merged: list[tuple[_Leaf, list[_Segment]]] = []
    i = 0
    while i < len(leaf_segments):
        leaf, segs = leaf_segments[i]
        total_tokens = sum(_approx_token_count(s.text) for s in segs)
        is_summary = leaf.heading_path and leaf.heading_path[-1].strip().lower() == "summary"
        is_last = i == len(leaf_segments) - 1

        if total_tokens < MERGE_SHORT_HEADING_TOKENS and not is_summary and not is_last:
            next_leaf, next_segs = leaf_segments[i + 1]
            if segs and next_segs and segs[-1].content_type == "prose" and next_segs[0].content_type == "prose":
                next_segs = [_Segment("prose", segs[-1].text + "\n\n" + next_segs[0].text)] + next_segs[1:]
                leaf_segments[i + 1] = (next_leaf, segs[:-1] + next_segs)
            else:
                leaf_segments[i + 1] = (next_leaf, segs + next_segs)
            i += 1
            continue
        if total_tokens < MERGE_SHORT_HEADING_TOKENS and not is_summary and is_last and merged:
            prev_leaf, prev_segs = merged[-1]
            merged[-1] = (prev_leaf, prev_segs + segs)
            i += 1
            continue

        merged.append((leaf, segs))
        i += 1

    chunks: list[Chunk] = []
    for leaf, segs in merged:
        for seg in segs:
            texts = [seg.text] if seg.content_type == "code" else _split_prose_to_limit(seg.text, max_tokens)
            for text in texts:
                if not text.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        chapter=chapter,
                        section=section,
                        heading_path=leaf.heading_path,
                        content_type=seg.content_type,
                        source_path=source_path,
                        text=text.strip(),
                        label=leaf.label,
                        has_math=_has_math(text) if seg.content_type == "prose" else False,
                    )
                )

    return chunks


def chunk_file(file_path: Path, root_dir: Path, **kwargs) -> list[Chunk]:
    """Chunk one `.md` file, deriving `chapter`/`section`/`source_path` from its location
    under `root_dir` (folder -> chapter, filename -> section). -> section 2, "Hierarchy
    levels to exploit"."""
    raw_markdown = file_path.read_text(encoding="utf-8")
    chapter = file_path.parent.name.removeprefix("chapter_")
    section = file_path.stem
    source_path = file_path.relative_to(root_dir).as_posix()
    return chunk_markdown(raw_markdown, chapter=chapter, section=section, source_path=source_path, **kwargs)


def walk_repo(root_dir: Path) -> list[Path]:
    """`chapter_*/**/*.md`, sorted for reproducible ordering. -> section 5, pipeline step 1."""
    return sorted(root_dir.glob("chapter_*/*.md"))


def chunk_repo(root_dir: str | Path, **kwargs) -> list[Chunk]:
    """Entry point: chunk every `.md` file in the corpus. -> section 5, pipeline steps 1-2."""
    root = Path(root_dir)
    chunks: list[Chunk] = []
    for file_path in walk_repo(root):
        chunks.extend(chunk_file(file_path, root, **kwargs))
    return chunks


if __name__ == "__main__":
    import sys
    from collections import Counter

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/d2l-en")
    all_chunks = chunk_repo(root)

    by_type = Counter(c.content_type for c in all_chunks)
    token_counts = [_approx_token_count(c.text) for c in all_chunks]
    print(f"files: {len(walk_repo(root))}")
    print(f"chunks: {len(all_chunks)} ({dict(by_type)})")
    print(f"avg tokens/chunk: {sum(token_counts) / max(len(token_counts), 1):.1f}")
    print(f"max tokens in a chunk: {max(token_counts, default=0)}")
    print(f"chunks with has_math=True: {sum(c.has_math for c in all_chunks)}")
    if all_chunks:
        print("\nsample chunk:")
        print(all_chunks[0].to_dict())
