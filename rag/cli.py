import sys

from rag.config import DEFAULT_QUERY


def query_from_argv() -> str:
    """The command-line words joined into one question, or the default question if none were given."""
    return " ".join(sys.argv[1:]) or DEFAULT_QUERY
