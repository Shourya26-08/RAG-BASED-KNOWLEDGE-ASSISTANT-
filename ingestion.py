from dataclasses import dataclass
from pathlib import Path
import re

from pypdf import PdfReader
from docx import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int

def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \\t]+", " ", text)
    text = re.sub(r"\\n{3,}", "\\n\\n", text)
    return text.strip()

def extract_text(path: str) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return _clean("\\n".join(page.extract_text() or "" for page in reader.pages))
    if suffix == ".docx":
        document = Document(str(file_path))
        return _clean("\\n".join(p.text for p in document.paragraphs))
    if suffix in {".txt", ".md"}:
        return _clean(file_path.read_text(encoding="utf-8", errors="ignore"))
    raise ValueError(f"Unsupported file type: {suffix}")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks

def build_chunks(paths: list[str], chunk_size: int, overlap: int) -> list[Chunk]:
    output = []
    for path in paths:
        text = extract_text(path)
        for index, part in enumerate(chunk_text(text, chunk_size, overlap)):
            output.append(Chunk(text=part, source=Path(path).name, chunk_id=index))
    return output
