"""
FFmpeg HTTP Service v0.2
FastAPI wrapper around ffmpeg for audio/video processing.

Endpoints:
  GET  /health              - Health check + ffmpeg version
  GET  /auth-test           - Auth verification
  POST /audio/cut-scenes    - Cut multiple scenes from a single audio file
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

app = FastAPI(title="FFmpeg Service", version="0.2.0")

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
    silence_threshold_db: float = -35.0,
    min_silence_duration: float = 0.4,
    max_silence_duration: float = 0.3,
    fade: float = 0.03
) -> dict:
    """
    Cut a single scene from input audio with silence removal and fades.
    Returns dict with scene data + base64 audio.
    """
    part = scene.get("part", "unknown")
    start = float(scene["start"])
    end = float(scene["end"])
    original_duration = end - start

    output_path = WORK_DIR / f"{job_id}_part_{part}.mp3"

    # Compute padded start/end
    seg_start = max(0, start - padding)
    seg_end = end + padding

    # Silence removal: pauses >= min_silence_duration get capped to max_silence_duration
    # Uses silenceremove filter with stop_periods=-1 (all internal silences)
    silence_filter = (
        f"silenceremove="
        f"start_periods=0:"
        f"stop_periods=-1:"
        f"stop_duration={max_silence_duration}:"
        f"stop_threshold={silence_threshold_db}dB"
    )

    # We need to know the duration AFTER silence removal to set afade out correctly
    # Two-pass approach: cut first, then apply fades

    # PASS 1: Cut + silence removal
    ret, err = run_ffmpeg([
        "-i", str(input_path),
        "-ss", f"{seg_start:.3f}",
        "-to", f"{seg_end:.3f}",
        "-af", silence_filter,
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(output_path)
    ])

    if ret != 0:
        return {
            "part": part,
            "status": "error",
            "error": f"ffmpeg cut failed: {err[:200]}"
        }

    # Get duration after silence removal
    duration_after_cut = get_duration(output_path)

    # PASS 2: Apply fade in/out (overwrite same file via temp)
    if duration_after_cut > 2 * fade:
        fade_out_start = duration_after_cut - fade
        fade_filter = f"afade=in:st=0:d={fade},afade=out:st={fade_out_start:.3f}:d={fade}"

        temp_path = WORK_DIR / f"{job_id}_part_{part}_faded.mp3"
        ret, err = run_ffmpeg([
            "-i", str(output_path),
            "-af", fade_filter,
            "-ar", "44100", "-ac", "2", "-b:a", "192k",
            str(temp_path)
        ])

        if ret == 0:
            # Swap files
            output_path.unlink(missing_ok=True)
            temp_path.rename(output_path)

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
        "silence_removed_ms": round((original_duration - final_duration) * 1000),
        "audio_base64": audio_base64,
        "size_bytes": len(audio_bytes)
    }


# ============================================
# ENDPOINTS
# ============================================
@app.get("/")
def root():
    return {"service": "ffmpeg", "version": "0.2.0", "status": "running"}


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
    silence_threshold_db: float = Form(-35.0),
    min_silence_duration: float = Form(0.4),
    max_silence_duration: float = Form(0.3),
    fade: float = Form(0.03),
    auth: str = Depends(verify_token)
):
    """
    Cut multiple scenes from a single combined audio file.

    Input:
      file: combined audio (mp3/wav) as multipart upload
      scenes: JSON string array of scenes, e.g.:
        [{"part": "1", "start": 0.0, "end": 2.612}, ...]

    Optional params:
      padding: buffer around each cut (default 0.05s = 50ms)
      silence_threshold_db: silence detection threshold (default -35dB)
      min_silence_duration: min pause length to trim (default 0.4s)
      max_silence_duration: cap long pauses to this (default 0.3s)
      fade: fade in/out duration (default 0.03s = 30ms)

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
                    silence_threshold_db,
                    min_silence_duration,
                    max_silence_duration,
                    fade
                )
                for scene in scenes_list
            ]
            results = [f.result() for f in futures]

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Count success/failure
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
        # Cleanup input
        input_path.unlink(missing_ok=True)
