# 
"""A simple chunker implemented for learning; production apps often use a library splitter.

For example, LangChain provides RecursiveCharacterTextSplitter, which tries to
split at natural text boundaries while respecting a configured chunk size.
"""


def chunk_text(
    text: str,
    max_chars: int = 1000,
    overlap_chars: int = 150,
) -> list[str]:
    # Work with whole words so chunks don't cut words in half.
    words = text.split()
    if not words:
        return []

    chunks = []
    chunk_start = 0

    while chunk_start < len(words):
        # 1. Build the current chunk up to max_chars
        chunk_end = chunk_start
        chunk_length = len(words[chunk_start])

        while chunk_end + 1 < len(words):
            next_word_len = len(words[chunk_end + 1]) + 1  # +1 for the connecting space
            if chunk_length + next_word_len > max_chars:
                break
            chunk_length += next_word_len
            chunk_end += 1

        # Move past the last included word so the slice includes it.
        chunk_end += 1
        chunks.append(" ".join(words[chunk_start:chunk_end]))

        # Stop if we've processed all words
        if chunk_end == len(words):
            break

        # 2. Look backward to calculate the overlap for the next chunk
        next_start = chunk_end
        overlap_length = 0

        while next_start > chunk_start:
            word_len = len(words[next_start - 1])
            if next_start < chunk_end:
                word_len += 1  # Include the space before this word

            if overlap_length + word_len > overlap_chars:
                break

            overlap_length += word_len
            next_start -= 1

        # Avoid repeating the entire chunk, so the next chunk always makes progress.
        chunk_start = next_start if next_start > chunk_start else chunk_end

    return chunks
