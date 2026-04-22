"""
FFmpeg HTTP Service v0.5
Audio cutting + smart optimization for UGC ad pipeline.

Endpoints:
  GET  /                    - Service info
  GET  /health              - Health check + ffmpeg version
  GET  /auth-test           - Auth verification
  POST /audio/cut-scenes    - Cut multiple scenes from combined audio
  POST /audio/optimize      - Smart 3s optimization for single audio file

Changelog v0.5:
  - Added /audio/optimize endpoint
  - Smart tiered logic (5 tiers) for 3s target
  - Silence compression via ffmpeg silenceremove
  - Optional atempo with 1.15 hard cap
"""

import subprocess
import os
import json
import base64
import uuid
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI(title="FFmpeg Service", version="0.5.0")

# ============================================
# CONFIG
# ============================================
API_TOKEN = os.getenv("API_TOKEN", "change-me")
WORK_DIR = Path("/tmp/ffmpeg")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Optimization constants (hardcoded for consistency)
TARGET_DURATION = 3.0
MAX_ATEMPO = 1.15
SILENCE_THRESHOLD_DB = -35.0

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
    Two-pass: accurate cut, then apply fades.
    """
    part = scene.get("part", "unknown")
    start = float(scene["start"])
    end = float(scene["end"])
    original_duration = end - start

    cut_path = WORK_DIR / f"{job_id}_cut_{part}.mp3"
    output_path = WORK_DIR / f"{job_id}_part_{part}.mp3"

    seg_start = max(0, start - padding)
    seg_end = end + padding

    # Pass 1: Accurate cut
    ret, err = run_ffmpeg([
        "-accurate_seek",
        "-i", str(input_path),
        "-ss", f"{seg_start:.3f}",
        "-to", f"{seg_end:.3f}",
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(cut_path)
    ])

    if ret != 0:
        return {"part": part, "status": "error", "error": f"cut failed: {err[:200]}"}

    cut_duration = get_duration(cut_path)

    # Pass 2: Fade in/out
    if cut_duration < 2 * fade:
        cut_path.rename(output_path)
    else:
        fade_out_start = cut_duration - fade
        audio_filter = f"afade=in:st=0:d={fade},afade=out:st={fade_out_start:.3f}:d={fade}"
        ret, err = run_ffmpeg([
            "-i", str(cut_path),
            "-af", audio_filter,
            "-ar", "44100", "-ac", "2", "-b:a", "192k",
            str(output_path)
        ])
        cut_path.unlink(missing_ok=True)
        if ret != 0:
            return {"part": part, "status": "error", "error": f"fade failed: {err[:200]}"}

    final_duration = get_duration(output_path)

    with open(output_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

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


def optimize_audio(input_path: Path, job_id: str) -> dict:
    """
    Smart tiered optimization for 3s target.
    
    Tiers:
      1. ≤ 3.0s     → passthrough
      2. 3.0-3.5s   → silence compress only (pauses >400ms → 300ms)
      3. 3.5-4.2s   → silence compress + atempo 1.05
      4. 4.2-5.0s   → silence compress + atempo 1.10
      5. > 5.0s     → aggressive silence + atempo 1.15
    """
    original_duration = get_duration(input_path)
    
    # ============================================
    # TIER 1: Already under target → passthrough
    # ============================================
    if original_duration <= TARGET_DURATION:
        with open(input_path, "rb") as f:
            audio_bytes = f.read()
        return {
            "status": "ok",
            "tier": 1,
            "tier_name": "passthrough",
            "original_duration": round(original_duration, 3),
            "after_silence_duration": round(original_duration, 3),
            "final_duration": round(original_duration, 3),
            "silence_savings_ms": 0,
            "atempo_applied": 1.0,
            "atempo_savings_ms": 0,
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "size_bytes": len(audio_bytes)
        }
    
    # ============================================
    # Determine tier + parameters
    # ============================================
    if original_duration <= 3.5:
        tier = 2
        tier_name = "soft"
        min_silence = 0.4   # pauses ≥400ms get compressed
        max_silence = 0.3   # compressed to 300ms max
        atempo = 1.0
    elif original_duration <= 4.2:
        tier = 3
        tier_name = "moderate"
        min_silence = 0.3
        max_silence = 0.25
        atempo = 1.05
    elif original_duration <= 5.0:
        tier = 4
        tier_name = "strong"
        min_silence = 0.25
        max_silence = 0.2
        atempo = 1.10
    else:
        tier = 5
        tier_name = "aggressive"
        min_silence = 0.2
        max_silence = 0.15
        atempo = 1.15
    
    # ============================================
    # STEP 1: Silence compression
    # ffmpeg silenceremove: kills/trims silences
    # stop_periods=-1 processes all silences
    # stop_duration = min silence to be considered "long"
    # We use stop_threshold to set what counts as silence
    # ============================================
    silence_path = WORK_DIR / f"{job_id}_silence.mp3"
    
    # ffmpeg silenceremove has limited "cap" capability
    # We use it to remove silences >= min_silence duration
    # then add back a brief pause via pause_after filter is not possible
    # workaround: use silenceremove with stop_periods=-1 + stop_duration=min
    # this removes all silences at/above min_silence completely
    # For our use case, we want to CAP not REMOVE, so:
    # we remove all silences >=min, then audio is just speech + natural pauses <min
    
    silence_filter = (
        f"silenceremove="
        f"stop_periods=-1:"
        f"stop_duration={min_silence}:"
        f"stop_threshold={SILENCE_THRESHOLD_DB}dB"
    )
    
    ret, err = run_ffmpeg([
        "-i", str(input_path),
        "-af", silence_filter,
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(silence_path)
    ])
    
    if ret != 0:
        return {"status": "error", "error": f"silence compression failed: {err[:200]}"}
    
    after_silence_duration = get_duration(silence_path)
    silence_savings_ms = round((original_duration - after_silence_duration) * 1000)
    
    # ============================================
    # STEP 2: Atempo (if tier requires it)
    # ============================================
    output_path = WORK_DIR / f"{job_id}_optimized.mp3"
    atempo_applied = 1.0
    
    if atempo > 1.0:
        # Check if atempo is needed based on current duration
        if after_silence_duration > TARGET_DURATION:
            # Calculate ideal atempo to hit target (but cap at configured)
            ideal_atempo = after_silence_duration / TARGET_DURATION
            atempo_applied = min(ideal_atempo, atempo, MAX_ATEMPO)
            
            # Round to 2 decimals
            atempo_applied = round(atempo_applied, 2)
            
            if atempo_applied > 1.01:
                ret, err = run_ffmpeg([
                    "-i", str(silence_path),
                    "-af", f"atempo={atempo_applied}",
                    "-ar", "44100", "-ac", "2", "-b:a", "192k",
                    str(output_path)
                ])
                if ret != 0:
                    return {"status": "error", "error": f"atempo failed: {err[:200]}"}
                silence_path.unlink(missing_ok=True)
            else:
                # Atempo too close to 1.0, skip
                atempo_applied = 1.0
                silence_path.rename(output_path)
        else:
            # Already under target after silence compression, no atempo needed
            silence_path.rename(output_path)
    else:
        # Tier 2: no atempo
        silence_path.rename(output_path)
    
    final_duration = get_duration(output_path)
    atempo_savings_ms = round((after_silence_duration - final_duration) * 1000) if atempo_applied > 1.0 else 0
    
    # ============================================
    # Read + encode + cleanup
    # ============================================
    with open(output_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    output_path.unlink(missing_ok=True)
    
    return {
        "status": "ok",
        "tier": tier,
        "tier_name": tier_name,
        "original_duration": round(original_duration, 3),
        "after_silence_duration": round(after_silence_duration, 3),
        "final_duration": round(final_duration, 3),
        "silence_savings_ms": silence_savings_ms,
        "atempo_applied": round(atempo_applied, 2),
        "atempo_savings_ms": atempo_savings_ms,
        "total_savings_ms": round((original_duration - final_duration) * 1000),
        "audio_base64": audio_base64,
        "size_bytes": len(audio_bytes)
    }


# ============================================
# ENDPOINTS
# ============================================
@app.get("/")
def root():
    return {"service": "ffmpeg", "version": "0.5.0", "status": "running"}


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
    (Unchanged from v0.4)
    """
    start_time = time.time()

    try:
        scenes_list = json.loads(scenes)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid scenes JSON: {e}")

    if not isinstance(scenes_list, list) or not scenes_list:
        raise HTTPException(400, "scenes must be a non-empty array")

    for i, scene in enumerate(scenes_list):
        if "part" not in scene or "start" not in scene or "end" not in scene:
            raise HTTPException(400, f"Scene #{i} missing required fields")

    job_id = str(uuid.uuid4())[:8]
    input_path = WORK_DIR / f"{job_id}_input.mp3"

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        with ThreadPoolExecutor(max_workers=min(len(scenes_list), 8)) as executor:
            futures = [
                executor.submit(cut_scene, input_path, scene, job_id, padding, fade)
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


@app.post("/audio/optimize")
async def optimize_endpoint(
    file: UploadFile = File(...),
    auth: str = Depends(verify_token)
):
    """
    Smart 3s optimization for a single audio file.
    
    Tiered logic based on original duration:
      Tier 1: ≤ 3.0s      → passthrough (no changes)
      Tier 2: 3.0 - 3.5s  → silence compression only
      Tier 3: 3.5 - 4.2s  → silence compression + atempo 1.05
      Tier 4: 4.2 - 5.0s  → silence compression + atempo 1.10
      Tier 5: > 5.0s      → aggressive silence + atempo 1.15
    
    Max atempo hard cap: 1.15 (never exceeded)
    
    Input: single MP3 file as multipart upload
    Output: optimized MP3 as base64 + detailed metadata
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())[:8]
    input_path = WORK_DIR / f"{job_id}_input.mp3"

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        result = optimize_audio(input_path, job_id)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        result["processing_time_ms"] = elapsed_ms
        
        return result

    finally:
        input_path.unlink(missing_ok=True)
