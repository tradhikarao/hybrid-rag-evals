from rag.documents import Hit


def print_hits(title: str, hits: list[Hit]) -> None:
    print(f"\n--- {title} ---")
    for rank, hit in enumerate(hits, start=1):
        preview = hit.text[:80].replace("\n", " ")
        print(f"{rank}. [{hit.score:.4f}] {hit.source}: {preview}...")
