from tempfile import NamedTemporaryFile
import os

import whisper
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

app = FastAPI(title="Whisper API")

API_KEY = os.getenv("WHISPER_API_KEY", "")
model = whisper.load_model("small")


def require_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/")
async def root():
    return {"message": "Whisper API is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    task: str = Form("transcribe"),
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    temp_path = None

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        options = {"task": task}
        if language:
            options["language"] = language

        result = model.transcribe(temp_path, **options)

        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language"),
            "segments": result.get("segments", []),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
