from typing import List, Tuple, Optional
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import hnswlib
import tempfile
import soundfile as sf
from pydub import AudioSegment

# ASR: faster-whisper
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

# LLM wrappers: gpt4all
GPT4ALL_AVAILABLE = False
try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
except Exception:
    GPT4ALL_AVAILABLE = False

# TTS: use pyttsx3 as fallback
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

# -----------------------
# Embeddings + vector DB
# -----------------------
class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384, index_path: str = "data/emb_index.bin", meta_path: str = "data/meta.json"):
        self.model = SentenceTransformer(model_name)
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load()
        else:
            # init empty index
            self.index.init_index(max_elements=100000, ef_construction=200, M=16)
            self.meta = []
            self.next_id = 0

    def _load(self):
        self.index.load_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.next_id = len(self.meta)

    def save(self):
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        self.index.save_index(self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def add_texts(self, texts: List[str], metadatas: List[dict]):
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        count = embs.shape[0]
        ids = list(range(self.next_id, self.next_id + count))
        self.index.add_items(embs, ids)
        for i, md in enumerate(metadatas):
            self.meta.append({"id": ids[i], **md})
        self.next_id += count
        self.save()

    def query(self, text: str, k: int = 6) -> List[dict]:
        emb = self.model.encode([text], convert_to_numpy=True)
        labels, distances = self.index.knn_query(emb, k=k)
        results = []
        for lid, dist in zip(labels[0], distances[0]):
            md = next((m for m in self.meta if m["id"] == int(lid)), None)
            if md:
                results.append({"score": float(dist), "metadata": md})
        return results

# -------------
# ASR wrapper
# -------------
class ASRWrapper:
    def __init__(self, model_dir: str = "backend/models", model_name: Optional[str] = None):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper not installed. See README.")
        self.model_dir = model_dir
        self.model_name = model_name
        model_file = model_name or self._find_model()
        if model_file is None:
            raise FileNotFoundError("No whisper model found in backend/models/. Place model file (e.g., 'whisper-small.bin').")
        self.model = WhisperModel(model_file, device="cpu", compute_type="int8")

    def _find_model(self):
        for f in os.listdir(self.model_dir):
            if "whisper" in f or f.endswith(".bin"):
                return os.path.join(self.model_dir, f)
        return None

    def transcribe(self, wav_path: str, language: Optional[str] = None) -> str:
        audio = AudioSegment.from_file(wav_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.set_frame_rate(16000).set_channels(1).export(tmp.name, format="wav")
        segments, info = self.model.transcribe(tmp.name, beam_size=5, language=language)
        text = "".join([s.text for s in segments])
        return text

# -------------
# LLM Wrapper (gpt4all)
# -------------
class LLMWrapper:
    def __init__(self, model_path: str = "backend/models/gpt4all.bin"):
        if not GPT4ALL_AVAILABLE:
            raise RuntimeError("gpt4all python package not installed. See README.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GPT4All model not found at {model_path}")
        self.model = GPT4All(model=model_path)

    def generate(self, system_prompt: str, messages: List[dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
        prompt_parts = [system_prompt]
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Kazuha: {content}")
        prompt_parts.append("Kazuha:")
        prompt = "\n\n".join(prompt_parts)
        out = self.model.generate(prompt)
        if isinstance(out, str):
            return out
        # some gpt4all versions return dict-like or other; attempt to coerce
        try:
            return str(out)
        except Exception:
            return ""

# -------------
# TTS Wrapper (pyttsx3)
# -------------
class TTSWrapper:
    def __init__(self):
        self.engine = None
        if PYTTSX3_AVAILABLE:
            self.engine = pyttsx3.init()

    def synthesize(self, text: str, out_path: str = "data/out.wav"):
        if self.engine is None:
            raise RuntimeError("pyttsx3 not available. Install it or add a Coqui TTS model.")
        # pyttsx3 save_to_file may not always be instantaneous; ensure directory exists
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.engine.save_to_file(text, out_path)
        self.engine.runAndWait()

# -------------
# Utilities
# -------------
def chunk_text(text: str, max_tokens: int = 512, overlap: int = 64) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + max_tokens, len(words))
        chunk = " ".join(words[i:j])
        chunks.append(chunk)
        i = j - overlap
        if i < 0:
            i = 0
    return chunks
