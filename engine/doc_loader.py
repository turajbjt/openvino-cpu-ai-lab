def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return chunk_text(text)


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
