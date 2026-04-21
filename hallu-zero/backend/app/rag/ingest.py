"""
Document ingestion — optimized for large files (40+ pages).
Small chunks (100 words) with overlap for precise retrieval.
"""
import asyncio
import argparse
from pathlib import Path

from app.rag.pipeline import Document, get_rag_pipeline


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """
    Small chunks = more precise retrieval on large docs.
    100 words per chunk with 20-word overlap.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def chunk_by_section(text: str) -> list[str]:
    """
    Also try splitting by headings/sections for structured docs like Power BI notes.
    Splits on lines that look like headers (short, no period at end).
    """
    lines = text.splitlines()
    sections = []
    current = []

    for line in lines:
        stripped = line.strip()
        # Detect heading: short line, no period, not empty
        is_heading = (
            stripped and
            len(stripped) < 80 and
            not stripped.endswith('.') and
            not stripped.endswith(',') and
            (stripped.isupper() or stripped.istitle() or stripped.endswith(':'))
        )
        if is_heading and current:
            section_text = " ".join(" ".join(current).split())
            if len(section_text.split()) > 10:
                sections.append(section_text)
            current = [stripped]
        else:
            if stripped:
                current.append(stripped)

    if current:
        section_text = " ".join(" ".join(current).split())
        if len(section_text.split()) > 10:
            sections.append(section_text)

    return sections


def load_file(path: Path) -> list[Document]:
    docs = []
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages_text = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append((page_num + 1, page_text))

            # Create per-page chunks AND word-level chunks
            all_chunks = []

            for page_num, page_text in pages_text:
                # Clean up
                clean = "\n".join(l for l in page_text.splitlines() if l.strip())

                # Word-level small chunks per page
                word_chunks = chunk_text(clean, chunk_size=100, overlap=20)
                for i, chunk in enumerate(word_chunks):
                    if chunk.strip():
                        all_chunks.append((chunk, {
                            "source": path.name,
                            "page": page_num,
                            "chunk": i,
                            "type": "word",
                        }))

                # Section-level chunks per page
                section_chunks = chunk_by_section(clean)
                for i, chunk in enumerate(section_chunks):
                    if chunk.strip():
                        all_chunks.append((chunk, {
                            "source": path.name,
                            "page": page_num,
                            "chunk": i,
                            "type": "section",
                        }))

            for content, metadata in all_chunks:
                docs.append(Document(content=content, metadata=metadata))

            return docs

        except ImportError:
            print(f"pypdf not installed, skipping {path}")
            return []

    elif suffix == ".json":
        import json
        data = json.loads(path.read_text())
        text = json.dumps(data, indent=2) if isinstance(data, dict) else str(data)
    else:
        return []

    # For txt/md/json — use both word chunks and section chunks
    clean = "\n".join(l for l in text.splitlines() if l.strip())

    word_chunks = chunk_text(clean, chunk_size=100, overlap=20)
    for i, chunk in enumerate(word_chunks):
        if chunk.strip():
            docs.append(Document(
                content=chunk,
                metadata={"source": path.name, "chunk": i, "type": "word"},
            ))

    section_chunks = chunk_by_section(clean)
    for i, chunk in enumerate(section_chunks):
        if chunk.strip():
            docs.append(Document(
                content=chunk,
                metadata={"source": path.name, "chunk": i, "type": "section"},
            ))

    return docs


async def ingest_directory(directory: str):
    pipeline = get_rag_pipeline()
    path = Path(directory)
    if not path.exists():
        print(f"Directory {directory} does not exist.")
        return

    all_docs = []
    for file in path.rglob("*"):
        if file.is_file() and file.suffix.lower() in {".txt", ".pdf", ".md", ".json"}:
            print(f"  Loading: {file.name}")
            docs = load_file(file)
            all_docs.extend(docs)
            print(f"    → {len(docs)} chunks")

    if not all_docs:
        print("No documents found.")
        return

    print(f"\nIngesting {len(all_docs)} chunks...")
    result = await pipeline.ingest(all_docs)
    print(f"Done. Total in store: {result['total']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./data/documents")
    args = parser.parse_args()
    asyncio.run(ingest_directory(args.path))