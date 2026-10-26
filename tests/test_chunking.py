from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_returns_one_chunk():
    assert chunk_text("These words fit.") == ["These words fit."]


def test_text_splits_at_maximum_size():
    chunks = chunk_text("aa bb cc", max_chars=5, overlap_chars=0)

    assert chunks == ["aa bb", "cc"]


def test_chunks_repeat_words_for_overlap():
    chunks = chunk_text("aa bb cc dd", max_chars=5, overlap_chars=2)

    assert chunks == ["aa bb", "bb cc", "cc dd"]