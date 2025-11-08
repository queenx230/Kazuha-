import os
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
from models import ASRWrapper, LLMWrapper, EmbeddingStore, TTSWrapper
from persona import get_system_prompt, USER_NAME
from ingest import main as run_ingest
import shutil
import uuid
import tempfile

app = FastAPI()

# serve frontend static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Initialize components (lazy)
ASR = None
LLM = None
EMB = None
TTS = None

CONV_HISTORY_PATH = "conv_history.json"
if not os.path.exists(CONV_HISTORY_PATH):
    with open(CONV_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f)

def load_components():
    global ASR, LLM, EMB, TTS
    if ASR is None:
        try:
            ASR = ASRWrapper(model_dir="models")
        except Exception as e:
            print("ASR init error:", e)
            ASR = None
    if EMB is None:
        EMB = EmbeddingStore(model_name="all-MiniLM-L6-v2", dim=384, index_path="data/emb_index.bin", meta_path="data/meta.json")
    if LLM is None:
        try:
            if os.path.exists("models/gpt4all.bin"):
                LLM = LLMWrapper(model_path="models/gpt4all.bin")
            else:
                mp = None
                for f in os.listdir("models"):
                    if f.endswith(".bin"):
                        mp = os.path.join("models", f)
                        break
                if mp:
                    LLM = LLMWrapper(model_path=mp)
                else:
                    print("No LLM model file found in backend/models/. Place a quantized model there.")
                    LLM = None
        except Exception as e:
            print("LLM init error:", e)
            LLM = None
    if TTS is None:
        TTS = TTSWrapper()
    return ASR, LLM, EMB, TTS

# Basic homepage serving the frontend index
@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/ingest")
async def ingest_endpoint():
    run_ingest()
    return {"status": "ok"}

@app.post("/chat_text")
async def chat_text(request: Request):
    payload = await request.json()
    user_text = payload.get("text", "")
    lang = payload.get("language", None)
    return await handle_conversation(user_text, lang, audio_requested=True)

@app.post("/chat_audio")
async def chat_audio(file: UploadFile = File(...), language: str = Form(None)):
    tmp_dir = tempfile.mkdtemp()
    try:
        file_path = os.path.join(tmp_dir, f"upload_{uuid.uuid4().hex}_{file.filename}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        load_components()
        if ASR is None:
            return JSONResponse({"error":"ASR engine not initialized. See README."}, status_code=500)
        transcript = ASR.transcribe(file_path, language=language)
        return await handle_conversation(transcript, language, audio_requested=True)
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

async def handle_conversation(user_text: str, language: str = None, audio_requested: bool = True):
    load_components()
    if LLM is None:
        return JSONResponse({"error":"LLM engine not initialized. Place a model in backend/models/ and install required packages."}, status_code=500)

    # retrieve RAG chunks
    contexts = []
    if EMB is not None:
        hits = EMB.query(user_text, k=5)
        for h in hits:
            md = h["metadata"]
            contexts.append(f"Source: {md.get('source','?')} - {md.get('preview','')}")
    # load conversation history per user (single local user Sonaf)
    with open(CONV_HISTORY_PATH, "r", encoding="utf-8") as f:
        conv = json.load(f)
    hist = conv.get(USER_NAME, [])
    # build messages: include last few messages
    messages = []
    for m in hist[-6:]:
        messages.append(m)
    messages.append({"role":"user","content":user_text})

    system_prompt = get_system_prompt(USER_NAME)
    if contexts:
        system_prompt = system_prompt + "\n\nContextual documents:\n" + "\n\n".join(contexts)

    assistant_text = LLM.generate(system_prompt, messages, max_tokens=512, temperature=0.6)

    hist.append({"role":"user","content":user_text})
    hist.append({"role":"assistant","content":assistant_text})
    conv[USER_NAME] = hist
    with open(CONV_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)

    audio_url = None
    if audio_requested:
        out_file = f"static_audio_{uuid.uuid4().hex}.wav"
        out_path = os.path.join("data", out_file)
        try:
            TTS.synthesize(assistant_text, out_path)
            audio_url = f"/audio/{out_file}"
        except Exception as e:
            print("TTS error:", e)
            audio_url = None

    return JSONResponse({"text": assistant_text, "audio_url": audio_url})

@app.get("/audio/{name}")
async def get_audio(name: str):
    path = os.path.join("data", name)
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav")
    return JSONResponse({"error":"file not found"}, status_code=404)

if __name__ == "__main__":
    load_components()
    uvicorn.run("main:app", host="127.0.0.1", port=7860, reload=True)
