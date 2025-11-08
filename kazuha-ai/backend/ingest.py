import os
from models import EmbeddingStore, chunk_text
from PyPDF2 import PdfReader

DATA_DIR = "data"
INDEX_BIN = "data/emb_index.bin"
META_JSON = "data/meta.json"

def extract_text_from_file(path: str) -> str:
    path_l = path.lower()
    if path_l.endswith(".txt") or path_l.endswith(".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif path_l.endswith(".pdf"):
        text = []
        reader = PdfReader(path)
        for p in reader.pages:
            try:
                text.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(text)
    else:
        return ""

def main():
    emb = EmbeddingStore(index_path=INDEX_BIN, meta_path=META_JSON)
    texts = []
    metas = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            fp = os.path.join(root, f)
            if f.lower().endswith(('.txt', '.md', '.pdf')):
                print("Reading", fp)
                content = extract_text_from_file(fp)
                if not content:
                    continue
                chunks = chunk_text(content, max_tokens=400, overlap=50)
                for idx, c in enumerate(chunks):
                    texts.append(c)
                    metas.append({"source": fp, "chunk_id": idx, "preview": c[:200]})
    if texts:
        emb.add_texts(texts, metas)
        print("Indexed", len(texts), "chunks.")
    else:
        print("No documents found to index in data/")

if __name__ == "__main__":
    main()
