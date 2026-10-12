from dataclasses import dataclass

from rag.config import DATA_DIR


@dataclass(frozen=True)
class Document:
    source: str  # file name, e.g. "01_pricing.txt"
    text: str


@dataclass(frozen=True)
class Hit:
    """One retrieved document plus the score the retrieval stage gave it."""

    index: int  # position in the document list
    document: Document
    score: float

    @property
    def source(self) -> str:
        return self.document.source

    @property
    def text(self) -> str:
        return self.document.text


def load_documents() -> list[Document]:
    paths = sorted(DATA_DIR.glob("*.txt"))  # sorted for a stable document order
    if not paths:
        raise RuntimeError(f"No .txt files found in {DATA_DIR}")
    return [Document(source=path.name, text=path.read_text().strip()) for path in paths]
