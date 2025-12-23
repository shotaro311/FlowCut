from src.llm.chunking import split_words_into_time_chunks
from src.transcribe.base import WordTimestamp


def test_split_words_into_time_chunks_prefers_gaps():
    words = [
        WordTimestamp(word="a", start=0.0, end=1.0),
        WordTimestamp(word="b", start=1.0, end=2.0),
        WordTimestamp(word="c", start=2.0, end=3.0),
        WordTimestamp(word="d", start=3.0, end=4.0),
        WordTimestamp(word="e", start=4.0, end=5.0),
        # gap=1.0s
        WordTimestamp(word="f", start=6.0, end=7.0),
        WordTimestamp(word="g", start=7.0, end=8.0),
    ]

    chunks = split_words_into_time_chunks(words, chunk_sec=5.0, snap_window_sec=2.0, min_gap_sec=0.2)

    assert [(c.start_idx, c.end_idx) for c in chunks] == [(0, 4), (5, 6)]

