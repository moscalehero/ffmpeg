import subprocess
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI(title="FFmpeg Service", version="0.1.0")

API_TOKEN = os.getenv("API_TOKEN", "change-me")
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.get("/")
def root():
    return {"service": "ffmpeg", "status": "running"}

@app.get("/health")
def health():
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        ffmpeg_v = r.stdout.split("\n")[0] if r.returncode == 0 else "ERROR"
    except Exception as e:
        ffmpeg_v = f"ERROR: {e}"
    return {"status": "ok", "ffmpeg": ffmpeg_v}

@app.get("/auth-test")
def auth_test(auth: str = Depends(verify_token)):
    return {"status": "authenticated"}
