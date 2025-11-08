Kazuha-AI — local, privacy-first personal assistant (Windows 11, CPU-only)

Quick summary
- Fully local: ASR, LLM, embeddings, vector DB, and TTS run on your laptop.
- MVP features: voice/text chat (Talk button), lively playful persona, Arabic/English support, local ingestion of long docs, animated avatar (cartoon/anime style), and local conversation history.
- You must download model binaries (LLM, ASR, and optionally TTS) and place them in kazuha-ai/backend/models/.

Requirements (high-level)
- Windows 11, Python 3.10+
- 16 GB RAM (you have it)
- Models: a GGML gpt4all model (recommended for Windows), and a whisper/faster-whisper model for ASR.

Install steps (PowerShell)
1) Clone repo and create venv
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2) Install Python packages
   pip install --upgrade pip
   pip install -r kazuha-ai/requirements.txt

3) (Torch for faster-whisper on CPU)
   pip install torch --index-url https://download.pytorch.org/whl/cpu

4) Download models (place in kazuha-ai/backend/models/)
   - ASR (faster-whisper multilingual): download a local whisper model compatible with faster-whisper (use small/base for CPU). Save with a recognizable filename like "whisper-small.bin".
   - LLM: download a GPT4All ggml quantized model for Windows (e.g., ggml-gpt4all-j.bin) and save as "gpt4all.bin".
   - TTS: pyttsx3 uses Windows built-in voices (no model download required). Optionally add Coqui TTS models into the models/ folder if you want higher-quality offline TTS.

5) Prepare data directory and conv history
   - Put PDFs, .txt, .md into kazuha-ai/backend/data/ before running ingest.
   - To secure conversation history file so only your Windows user can access it run (PowerShell as admin):
       icacls .\\kazuha-ai\\backend\\conv_history.json /inheritance:r /grant %USERNAME%:R

6) Start the server (from repository root)
   .\.venv\Scripts\Activate.ps1
   cd kazuha-ai\\backend
   uvicorn main:app --host 127.0.0.1 --port 7860 --reload

7) Open UI
   Open browser: http://127.0.0.1:7860/

Ingestion (index local docs)
- Put PDFs, .txt, .md into kazuha-ai/backend/data/
- Run: python ingest.py
  This converts docs -> chunks -> embeddings -> local hnswlib index (kazua-ai/backend/data/), ready for RAG.

Security & privacy
- No outgoing network requests at runtime except manual model downloads you perform. Validate model downloads yourself.
- All conversation history and indexes live in kazuha-ai/backend/data/ and backend/conv_history.json.
- The README recommends using icacls to restrict access to conv_history.json.

---
