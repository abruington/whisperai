from fastapi import FastAPI
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from tempfile import NamedTemporaryFile\
import os\
\
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header\
import whisper\
\
app = FastAPI(title="Whisper API")\
\
API_KEY = os.getenv("WHISPER_API_KEY", "")\
\
# "small" is a good starting point\
model = whisper.load_model("small")\
\
\
def require_api_key(x_api_key: str | None):\
    if API_KEY and x_api_key != API_KEY:\
        raise HTTPException(status_code=401, detail="Unauthorized")\
\
\
@app.get("/")\
async def root():\
    return \{"message": "Whisper API is running"\}\
\
\
@app.get("/health")\
async def health():\
    return \{"status": "ok"\}\
\
\
@app.post("/transcribe")\
async def transcribe_audio(\
    file: UploadFile = File(...),\
    language: str | None = Form(None),\
    task: str = Form("transcribe"),\
    x_api_key: str | None = Header(default=None),\
):\
    require_api_key(x_api_key)\
\
    suffix = os.path.splitext(file.filename)[1] or ".wav"\
    temp_path = None\
\
    try:\
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:\
            content = await file.read()\
            tmp.write(content)\
            temp_path = tmp.name\
\
        options = \{"task": task\}\
        if language:\
            options["language"] = language\
\
        result = model.transcribe(temp_path, **options)\
\
        return \{\
            "text": result.get("text", "").strip(),\
            "language": result.get("language"),\
            "segments": result.get("segments", []),\
        \}\
\
    except Exception as e:\
        raise HTTPException(status_code=500, detail=str(e))\
    finally:\
        if temp_path and os.path.exists(temp_path):\
            os.remove(temp_path)}
