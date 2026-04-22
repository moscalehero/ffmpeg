"""
FFmpeg HTTP Service v0.3
Clean audio cutting at exact timestamps.

Endpoints:
  GET  /                    - Service info
  GET  /health              - Health check + ffmpeg version
  GET  /auth-test           - Auth verification
  POST /audio/cut-scenes    - Cut multiple scenes from a single audio file

Changelog v0.3:
  - Removed silence removal (word-level timestamps are already precise)
  - Removed two-pass processing
  - Simpler + faster (single ffmpeg call per scene)
  - Keeps small fade in/out (30ms) for clean cut edges
"""

import subprocess
import os
import json
import base64
import uuid
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI(title="FFmpeg Service", version="0.3.0")

# ============================================
# CONFIG
# ============================================
API_TOKEN = os.getenv("API_TOKEN", "change-me")
WORK_DIR = Path("/tmp/ffmpeg")
WORK_DIR.mkdir(parents=True, exist_ok=True)

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials


# ============================================
# HELPERS
# ============================================
def run_ffmpeg(args: list) -> tuple:
    """Run ffmpeg command, return (returncode, stderr)."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.returncode, result.stderr


def get_duration(filepath: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(filepath)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def cut_scene(
    input_path: Path,
    scene: dict,
    job_id: str,
    padding: float = 0.05,
    fade: float = 0.03
) -> dict:
    """
    Cut a scene from input audio at exact timestamps.
    
    No silence removal - ElevenLabs word-level timestamps are already precise.
    Only adds small padding + fade in/out for clean cut edges (no clicks).
    """
    part = scene.get("part", "unknown")
    start = float(scene["start"])
    end = float(scene["end"])
    original_duration = end - start

    output_path = WORK_DIR / f"{job_id}_part_{part}.mp3"

    # Cut boundaries with small padding for breathing room
    seg_start = max(0, start - padding)
    seg_end = end + padding
    seg_duration = seg_end - seg_start

    # Fade in/out for clean cut edges (no clicks/pops)
    fade_out_start = max(0, seg_duration - fade)
    audio_filter = (
        f"afade=in:st=0:d={fade},"
        f"afade=out:st={fade_out_start:.3f}:d={fade}"
    )

    # Single ffmpeg pass: cut + fade
    ret, err = run_ffmpeg([
        "-i", str(input_path),
        "-ss", f"{seg_start:.3f}",
        "-to", f"{seg_end:.3f}",
        "-af", audio_filter,
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(output_path)
    ])

    if ret != 0:
        return {
            "part": part,
            "status": "error",
            "error": f"ffmpeg failed: {err[:200]}"
        }

    final_duration = get_duration(output_path)

    # Read + encode
    with open(output_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Cleanup
    output_path.unlink(missing_ok=True)

    return {
        "part": part,
        "status": "ok",
        "original_duration": round(original_duration, 3),
        "final_duration": round(final_duration, 3),
        "padding_added_ms": round(padding * 2 * 1000),
        "fade_ms": round(fade * 1000),
        "audio_base64": audio_base64,
        "size_bytes": len(audio_bytes)
    }


# ============================================
# ENDPOINTS
# ============================================
@app.get("/")
def root():
    return {"service": "ffmpeg", "version": "0.3.0", "status": "running"}


@app.get("/health")
def health():
    ffmpeg_v = "NOT FOUND"
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            ffmpeg_v = r.stdout.split("\n")[0]
    except Exception as e:
        ffmpeg_v = f"ERROR: {e}"
    return {"status": "ok", "ffmpeg": ffmpeg_v}


@app.get("/auth-test")
def auth_test(auth: str = Depends(verify_token)):
    return {"status": "authenticated"}


@app.post("/audio/cut-scenes")
async def cut_scenes(
    file: UploadFile = File(...),
    scenes: str = Form(...),
    padding: float = Form(0.05),
    fade: float = Form(0.03),
    auth: str = Depends(verify_token)
):
    """
    Cut multiple scenes from a single audio file at exact timestamps.

    Input:
      file: combined audio (mp3/wav) as multipart upload
      scenes: JSON string array of scenes, e.g.:
        [{"part": "1", "start": 0.0, "end": 2.612}, ...]

    Optional params:
      padding: buffer around each cut in seconds (default 0.05 = 50ms)
      fade: fade in/out duration in seconds (default 0.03 = 30ms)

    Returns:
      JSON with cut scenes as base64-encoded MP3s
    """
    start_time = time.time()

    # Parse scenes JSON
    try:
        scenes_list = json.loads(scenes)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid scenes JSON: {e}")

    if not isinstance(scenes_list, list) or not scenes_list:
        raise HTTPException(400, "scenes must be a non-empty array")

    # Validate each scene has required fields
    for i, scene in enumerate(scenes_list):
        if "part" not in scene or "start" not in scene or "end" not in scene:
            raise HTTPException(400, f"Scene #{i} missing required fields (part, start, end)")

    # Save input file
    job_id = str(uuid.uuid4())[:8]
    input_path = WORK_DIR / f"{job_id}_input.mp3"

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # Process all scenes in parallel
        with ThreadPoolExecutor(max_workers=min(len(scenes_list), 8)) as executor:
            futures = [
                executor.submit(
                    cut_scene,
                    input_path,
                    scene,
                    job_id,
                    padding,
                    fade
                )
                for scene in scenes_list
            ]
            results = [f.result() for f in futures]

        elapsed_ms = int((time.time() - start_time) * 1000)

        success_count = sum(1 for r in results if r.get("status") == "ok")
        error_count = sum(1 for r in results if r.get("status") == "error")

        return {
            "status": "ok" if error_count == 0 else "partial",
            "total_scenes": len(scenes_list),
            "success_count": success_count,
            "error_count": error_count,
            "total_processing_time_ms": elapsed_ms,
            "scenes": results
        }

    finally:
        input_path.unlink(missing_ok=True)
