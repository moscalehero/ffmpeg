"""
FFmpeg HTTP Service v0.9
Smart audio cutting + context-aware optimization + video optimization for UGC ad pipeline.

Endpoints:
  GET  /                    - Service info
  GET  /health              - Health check + ffmpeg version
  GET  /auth-test           - Auth verification
  POST /audio/cut-scenes    - Cut multiple scenes from combined audio
  POST /audio/optimize      - Smart optimization (target 3s, hard cap 4s, edge silence trim)
  POST /video/analyze       - Analyze video's audio track (NEW v0.9)
  POST /video/smart-cut     - Auto-trim leading/trailing silence from video (NEW v0.9)
  POST /video/speedup       - Speed up video + audio with constant pitch (NEW v0.9)
  POST /video/optimize      - Combined: smart-cut + auto-speedup if slow speaker (NEW v0.9)

Changelog v0.9:
  - NEW: Video endpoints for post-Seedance processing
  - /video/analyze: extract audio track and run full analysis
  - /video/smart-cut: trim leading/trailing silence from video (preserves mid-speech)
  - /video/speedup: constant-pitch speed change using setpts + atempo
  - /video/optimize: combined pipeline with auto speed recommendation based on speech style
  - All video endpoints handle video+audio in sync (setpts for video, atempo for audio)
  - REMOVED: pad_to_duration parameter from /audio/optimize
    (padding approach replaced by post-Seedance video smart-cut + speedup pipeline)

Changelog v0.8.1:
  - Auto-trim leading/trailing silence in /audio/optimize
  - Mid-speech pauses still use style-based rules (preserves natural breath)
  - 2% safety margin on atempo calculation

Changelog v0.7:
  - Removed /audio/analyze endpoint (analysis bundled inside /audio/optimize)
  - Style-scaled atempo caps: Fast 1.08, Normal 1.12, Slow 1.15
"""

import subprocess
import os
import json
import base64
import uuid
import time
import re
import shutil
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI(title="FFmpeg Service", version="0.9.0")

# ============================================
# CONFIG
# ============================================
API_TOKEN = os.getenv("API_TOKEN", "change-me")
WORK_DIR = Path("/tmp/ffmpeg")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Optimization targets
SOFT_TARGET = 3.0          # ideal duration
HARD_CAP = 4.0             # absolute max (never exceed)

# Atempo limits
ATEMPO_MAX = 1.15
ATEMPO_EMERGENCY = 1.20

# Style-scaled atempo caps (for duration between SOFT_TARGET and HARD_CAP)
ATEMPO_CAP_FAST = 1.08
ATEMPO_CAP_NORMAL = 1.12
ATEMPO_CAP_SLOW = 1.15

# Silence detection
SILENCE_RELATIVE_DB = 20
SILENCE_FALLBACK_DB = -35
MIN_PAUSE_DETECT_MS = 80

# Pause categorization (milliseconds)
PAUSE_MICRO_MAX = 180
PAUSE_SHORT_MAX = 300
PAUSE_MEDIUM_MAX = 500
PAUSE_LONG_MAX = 800

# Speech density thresholds
DENSITY_FAST = 0.85
DENSITY_NORMAL = 0.70

# Edge silence (v0.8.1) - always trim leading/trailing silence
EDGE_SILENCE_MIN_MS = 50      # trim edges only if they exceed this
EDGE_SILENCE_KEEP_MS = 50     # but keep this much for natural fade-in/out
EDGE_TOLERANCE = 0.05         # tolerance for detecting edge position (50ms)

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials


# ============================================
# BASIC HELPERS
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


# ============================================
# AUDIO ANALYSIS
# ============================================
def detect_volume(input_path: Path) -> dict:
    """Detect mean and max volume using volumedetect filter."""
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", str(input_path),
        "-af", "volumedetect",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    mean_db = None
    max_db = None
    
    for line in result.stderr.split("\n"):
        mean_match = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", line)
        max_match = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", line)
        if mean_match:
            mean_db = float(mean_match.group(1))
        if max_match:
            max_db = float(max_match.group(1))
    
    return {
        "mean_db": round(mean_db, 2) if mean_db is not None else None,
        "max_db": round(max_db, 2) if max_db is not None else None
    }


def detect_silences(input_path: Path, threshold_db: float, min_duration: float = 0.08) -> list:
    """Detect all silence periods in audio."""
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", str(input_path),
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    silences = []
    current_start = None
    
    for line in result.stderr.split("\n"):
        start_match = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        end_match = re.search(r"silence_end:\s*([\d.]+)", line)
        dur_match = re.search(r"silence_duration:\s*([\d.]+)", line)
        
        if start_match:
            current_start = float(start_match.group(1))
            if current_start < 0:
                current_start = 0.0
        elif end_match and current_start is not None:
            end_time = float(end_match.group(1))
            duration = float(dur_match.group(1)) if dur_match else (end_time - current_start)
            silences.append({
                "start": round(current_start, 3),
                "end": round(end_time, 3),
                "duration": round(duration, 3)
            })
            current_start = None
    
    return silences


def categorize_pause(duration_ms: int) -> str:
    """Categorize a pause by duration."""
    if duration_ms < PAUSE_MICRO_MAX:
        return "micro"
    elif duration_ms < PAUSE_SHORT_MAX:
        return "short"
    elif duration_ms < PAUSE_MEDIUM_MAX:
        return "medium"
    elif duration_ms < PAUSE_LONG_MAX:
        return "long"
    else:
        return "very_long"


def classify_speech_style(density: float) -> tuple:
    """Classify speech style based on density."""
    if density >= DENSITY_FAST:
        return "fast", f"high density ({density:.2f}), dense delivery with few pauses"
    elif density >= DENSITY_NORMAL:
        return "normal", f"balanced density ({density:.2f}), natural speech rhythm"
    else:
        return "slow", f"low density ({density:.2f}), many or long pauses"


def get_style_atempo_cap(style: str) -> float:
    """Get atempo cap based on speech style."""
    if style == "fast":
        return ATEMPO_CAP_FAST
    elif style == "normal":
        return ATEMPO_CAP_NORMAL
    else:  # slow
        return ATEMPO_CAP_SLOW


def analyze_audio(input_path: Path) -> dict:
    """Full analysis of an audio file."""
    duration = get_duration(input_path)
    volume = detect_volume(input_path)
    
    if volume["mean_db"] is not None:
        silence_threshold_db = volume["mean_db"] - SILENCE_RELATIVE_DB
    else:
        silence_threshold_db = SILENCE_FALLBACK_DB
    
    silences = detect_silences(
        input_path, 
        threshold_db=silence_threshold_db,
        min_duration=MIN_PAUSE_DETECT_MS / 1000
    )
    
    for s in silences:
        s["duration_ms"] = round(s["duration"] * 1000)
        s["category"] = categorize_pause(s["duration_ms"])
    
    total_silence = sum(s["duration"] for s in silences)
    active_speech = max(0, duration - total_silence)
    speech_density = active_speech / duration if duration > 0 else 0
    
    style, style_reasoning = classify_speech_style(speech_density)
    
    distribution = {"micro": 0, "short": 0, "medium": 0, "long": 0, "very_long": 0}
    for s in silences:
        distribution[s["category"]] += 1
    
    durations_ms = [s["duration_ms"] for s in silences]
    pause_stats = {
        "count": len(silences),
        "total_ms": round(total_silence * 1000),
        "avg_ms": round(sum(durations_ms) / len(durations_ms)) if durations_ms else 0,
        "min_ms": min(durations_ms) if durations_ms else 0,
        "max_ms": max(durations_ms) if durations_ms else 0,
        "distribution": distribution
    }
    
    recommendation = generate_recommendation(
        duration=duration,
        style=style,
        silences=silences,
        speech_density=speech_density
    )
    
    return {
        "duration": round(duration, 3),
        "volume": volume,
        "silence_threshold_db_used": round(silence_threshold_db, 2),
        "speech": {
            "active_speech_time": round(active_speech, 3),
            "total_silence_time": round(total_silence, 3),
            "speech_density": round(speech_density, 3),
            "style": style,
            "style_reasoning": style_reasoning
        },
        "pauses": {
            **pause_stats,
            "all_pauses": silences
        },
        "recommendation": recommendation
    }


# ============================================
# STRATEGY ENGINE
# ============================================
def get_trim_rules(style: str) -> dict:
    """Get pause trim caps based on speech style (in ms)."""
    if style == "fast":
        return {
            "micro": None, "short": None, "medium": None,
            "long": 300, "very_long": 350
        }
    elif style == "normal":
        return {
            "micro": None, "short": None, "medium": 250,
            "long": 200, "very_long": 250
        }
    else:  # slow
        return {
            "micro": None, "short": 180, "medium": 180,
            "long": 150, "very_long": 200
        }


def calculate_silence_savings(silences: list, trim_rules: dict) -> int:
    """Calculate how much silence (ms) would be saved."""
    savings_ms = 0
    for s in silences:
        cap = trim_rules.get(s["category"])
        if cap is not None and s["duration_ms"] > cap:
            savings_ms += s["duration_ms"] - cap
    return savings_ms


def generate_recommendation(duration: float, style: str, silences: list, speech_density: float) -> dict:
    """Generate optimization strategy recommendation."""
    if duration <= SOFT_TARGET:
        return {
            "needs_optimization": False,
            "reason": f"duration {duration:.3f}s already under target {SOFT_TARGET}s",
            "feasibility": "not_needed",
            "estimated_final_duration": duration
        }
    
    trim_rules = get_trim_rules(style)
    silence_savings_ms = calculate_silence_savings(silences, trim_rules)
    after_silence_s = duration - (silence_savings_ms / 1000)
    
    delta_to_target = max(0, after_silence_s - SOFT_TARGET)
    delta_to_cap = max(0, after_silence_s - HARD_CAP)
    style_cap = get_style_atempo_cap(style)
    
    if delta_to_cap <= 0:
        if delta_to_target <= 0:
            estimated_atempo = 1.0
        else:
            ideal = after_silence_s / SOFT_TARGET
            estimated_atempo = min(ideal, style_cap)
    else:
        required = after_silence_s / HARD_CAP
        ideal = after_silence_s / SOFT_TARGET
        
        if required <= ATEMPO_MAX:
            estimated_atempo = min(ideal, ATEMPO_MAX)
        elif required <= ATEMPO_EMERGENCY:
            estimated_atempo = required
        else:
            estimated_atempo = ATEMPO_EMERGENCY
    
    estimated_final = after_silence_s / estimated_atempo if estimated_atempo > 1.0 else after_silence_s
    hit_target = estimated_final <= SOFT_TARGET + 0.1
    hit_cap = estimated_final <= HARD_CAP + 0.05
    
    feasibility = "easy" if hit_target else ("moderate" if hit_cap else "hard")
    
    return {
        "needs_optimization": True,
        "feasibility": feasibility,
        "silence_savings_potential_ms": silence_savings_ms,
        "after_silence_duration": round(after_silence_s, 3),
        "estimated_atempo": round(estimated_atempo, 3),
        "estimated_final_duration": round(estimated_final, 3),
        "hit_soft_target": hit_target,
        "hit_hard_cap": hit_cap,
        "strategy": {
            "trim_rules_ms": trim_rules,
            "max_atempo": round(estimated_atempo, 3)
        }
    }


# ============================================
# OPTIMIZATION EXECUTION
# ============================================
def build_segments_from_silences(duration: float, silences: list, trim_rules: dict) -> tuple:
    """
    Build list of segments to keep/trim.
    
    v0.8.1: Edge silences (leading/trailing) are handled specially:
      - Leading silence (pause at start): trimmed to EDGE_SILENCE_KEEP_MS
      - Trailing silence (pause at end): trimmed to EDGE_SILENCE_KEEP_MS
      - Mid-speech pauses: use style-based trim_rules (unchanged)
    
    Returns:
        (segments, edge_info) tuple
    """
    segments = []
    last_end = 0.0
    
    # Identify leading and trailing silences (if any)
    leading_silence = None
    trailing_silence = None
    
    if silences:
        # Leading: pause starts at or very near 0
        if silences[0]["start"] <= EDGE_TOLERANCE and silences[0]["duration_ms"] > EDGE_SILENCE_MIN_MS:
            leading_silence = silences[0]
        
        # Trailing: pause ends at or very near duration
        # Don't double-count if same silence is both leading and trailing (very short edge case)
        if silences[-1]["end"] >= duration - EDGE_TOLERANCE and silences[-1]["duration_ms"] > EDGE_SILENCE_MIN_MS:
            if silences[-1] is not leading_silence:
                trailing_silence = silences[-1]
    
    edge_info = {
        "leading_trimmed_ms": 0,
        "trailing_trimmed_ms": 0,
        "leading_detected": leading_silence is not None,
        "trailing_detected": trailing_silence is not None
    }
    
    # Handle leading silence: start speech after leading silence (keep EDGE_SILENCE_KEEP_MS buffer)
    if leading_silence is not None:
        keep_seconds = EDGE_SILENCE_KEEP_MS / 1000
        last_end = max(0, leading_silence["end"] - keep_seconds)
        edge_info["leading_trimmed_ms"] = leading_silence["duration_ms"] - EDGE_SILENCE_KEEP_MS
    
    # Process mid-speech silences with style rules (skip edges - handled separately)
    for s in silences:
        if s is leading_silence or s is trailing_silence:
            continue
        
        cap_ms = trim_rules.get(s["category"])
        
        if cap_ms is None or s["duration_ms"] <= cap_ms:
            continue
        else:
            if s["start"] > last_end:
                segments.append(("speech", last_end, s["start"]))
            segments.append(("silence", s["start"], s["start"] + cap_ms / 1000))
            last_end = s["end"]
    
    # Handle trailing silence: cut before it, keep small buffer for natural fade
    if trailing_silence is not None:
        speech_end = trailing_silence["start"] + (EDGE_SILENCE_KEEP_MS / 1000)
        if speech_end > last_end:
            segments.append(("speech", last_end, speech_end))
        edge_info["trailing_trimmed_ms"] = trailing_silence["duration_ms"] - EDGE_SILENCE_KEEP_MS
    else:
        # No trailing silence - keep rest of audio
        if last_end < duration:
            segments.append(("speech", last_end, duration))
    
    # Safety fallback
    if not segments:
        segments.append(("speech", 0, duration))
    
    return segments, edge_info


def apply_silence_trim(input_path: Path, output_path: Path, segments: list) -> tuple:
    """Apply selective silence trimming using filter_complex."""
    # Single-segment case
    if len(segments) == 1 and segments[0][0] == "speech":
        seg_type, seg_start, seg_end = segments[0]
        input_duration = get_duration(input_path)
        # If covers whole file, just copy
        if seg_start <= 0.01 and seg_end >= input_duration - 0.01:
            shutil.copy(input_path, output_path)
            return 0, ""
        # Otherwise trim to the subset (handles edge-only trimming cases)
        return run_ffmpeg([
            "-i", str(input_path),
            "-af", f"atrim=start={seg_start:.3f}:end={seg_end:.3f},asetpts=PTS-STARTPTS",
            "-ar", "44100", "-ac", "2", "-b:a", "192k",
            str(output_path)
        ])
    
    filter_parts = []
    concat_inputs = []
    
    for i, (seg_type, seg_start, seg_end) in enumerate(segments):
        duration = seg_end - seg_start
        if duration <= 0:
            continue
        
        if seg_type == "speech":
            filter_parts.append(
                f"[0:a]atrim=start={seg_start:.3f}:end={seg_end:.3f},"
                f"asetpts=PTS-STARTPTS[a{i}]"
            )
        else:
            filter_parts.append(
                f"anullsrc=r=44100:cl=stereo:d={duration:.3f}[a{i}]"
            )
        concat_inputs.append(f"[a{i}]")
    
    if not concat_inputs:
        shutil.copy(input_path, output_path)
        return 0, ""
    
    filter_complex = (
        ";".join(filter_parts) + ";" +
        "".join(concat_inputs) +
        f"concat=n={len(concat_inputs)}:v=0:a=1[out]"
    )
    
    return run_ffmpeg([
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(output_path)
    ])


def apply_atempo(input_path: Path, output_path: Path, tempo: float) -> tuple:
    """Apply atempo filter."""
    return run_ffmpeg([
        "-i", str(input_path),
        "-af", f"atempo={tempo:.3f}",
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(output_path)
    ])


def optimize_audio(input_path: Path, job_id: str) -> dict:
    """Full optimization pipeline: analyze + context-aware execution."""
    # STEP 1: Analyze
    analysis = analyze_audio(input_path)
    duration = analysis["duration"]
    style = analysis["speech"]["style"]
    silences = analysis["pauses"]["all_pauses"]
    
    # STEP 2: Short file passthrough path
    if duration <= SOFT_TARGET:
        passthrough_path = WORK_DIR / f"{job_id}_passthrough.mp3"
        shutil.copy(input_path, passthrough_path)
        
        with open(passthrough_path, "rb") as f:
            audio_bytes = f.read()
        passthrough_path.unlink(missing_ok=True)
        
        return {
            "status": "ok",
            "action": "passthrough",
            "analysis": analysis,
            "execution": {
                "silence_applied": False,
                "atempo_applied": 1.0,
                "silence_savings_ms": 0,
                "atempo_savings_ms": 0,
                "edge_info": {
                    "leading_trimmed_ms": 0,
                    "trailing_trimmed_ms": 0,
                    "leading_detected": False,
                    "trailing_detected": False
                }
            },
            "original_duration": round(duration, 3),
            "content_duration": round(duration, 3),
            "final_duration": round(duration, 3),
            "total_savings_ms": 0,
            "hit_soft_target": True,
            "hit_hard_cap": True,
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "size_bytes": len(audio_bytes)
        }
    
    # STEP 3: Apply silence trimming (v0.8.1 - includes edge silence auto-trim)
    trim_rules = get_trim_rules(style)
    segments, edge_info = build_segments_from_silences(duration, silences, trim_rules)
    
    silence_path = WORK_DIR / f"{job_id}_silence.mp3"
    ret, err = apply_silence_trim(input_path, silence_path, segments)
    
    if ret != 0:
        return {"status": "error", "error": f"silence trim failed: {err[:200]}"}
    
    after_silence_duration = get_duration(silence_path)
    silence_savings_ms = round((duration - after_silence_duration) * 1000)
    
    # STEP 4: Decide atempo (with 2% safety margin for ffmpeg approximation)
    atempo_applied = 1.0
    optimized_path = WORK_DIR / f"{job_id}_optimized.mp3"
    
    if after_silence_duration > HARD_CAP:
        # 2% safety margin: ffmpeg atempo actual output is ~1-2% off from mathematical target
        required = (after_silence_duration / HARD_CAP) * 1.02
        ideal = after_silence_duration / SOFT_TARGET
        
        if required <= ATEMPO_MAX:
            atempo_applied = min(ideal, ATEMPO_MAX)
        else:
            atempo_applied = min(required, ATEMPO_EMERGENCY)
        
        atempo_applied = round(atempo_applied, 3)
        
        ret, err = apply_atempo(silence_path, optimized_path, atempo_applied)
        silence_path.unlink(missing_ok=True)
        if ret != 0:
            return {"status": "error", "error": f"atempo failed: {err[:200]}"}
    
    elif after_silence_duration > SOFT_TARGET:
        style_cap = get_style_atempo_cap(style)
        ideal = after_silence_duration / SOFT_TARGET
        atempo_applied = round(min(ideal, style_cap), 3)
        
        if atempo_applied > 1.02:
            ret, err = apply_atempo(silence_path, optimized_path, atempo_applied)
            silence_path.unlink(missing_ok=True)
            if ret != 0:
                return {"status": "error", "error": f"atempo failed: {err[:200]}"}
        else:
            atempo_applied = 1.0
            silence_path.rename(optimized_path)
    else:
        silence_path.rename(optimized_path)
    
    content_duration = get_duration(optimized_path)
    atempo_savings_ms = round((after_silence_duration - content_duration) * 1000) if atempo_applied > 1.0 else 0
    
    final_duration = get_duration(optimized_path)
    
    with open(optimized_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    optimized_path.unlink(missing_ok=True)
    
    return {
        "status": "ok",
        "action": "optimized",
        "analysis": analysis,
        "execution": {
            "silence_applied": silence_savings_ms > 0,
            "atempo_applied": round(atempo_applied, 3),
            "silence_savings_ms": silence_savings_ms,
            "atempo_savings_ms": atempo_savings_ms,
            "trim_rules_used": trim_rules,
            "style_atempo_cap": round(get_style_atempo_cap(style), 3),
            "segments_count": len(segments),
            "edge_info": edge_info
        },
        "original_duration": round(duration, 3),
        "after_silence_duration": round(after_silence_duration, 3),
        "content_duration": round(content_duration, 3),
        "final_duration": round(final_duration, 3),
        "total_savings_ms": round((duration - content_duration) * 1000),
        "hit_soft_target": content_duration <= SOFT_TARGET + 0.1,
        "hit_hard_cap": content_duration <= HARD_CAP + 0.05,
        "audio_base64": audio_base64,
        "size_bytes": len(audio_bytes)
    }


# ============================================
# CUT SCENE
# ============================================
def cut_scene(
    input_path: Path,
    scene: dict,
    job_id: str,
    padding: float = 0.05,
    fade: float = 0.03
) -> dict:
    """Cut a scene from input audio at exact timestamps."""
    part = scene.get("part", "unknown")
    start = float(scene["start"])
    end = float(scene["end"])
    original_duration = end - start

    cut_path = WORK_DIR / f"{job_id}_cut_{part}.mp3"
    output_path = WORK_DIR / f"{job_id}_part_{part}.mp3"

    seg_start = max(0, start - padding)
    seg_end = end + padding

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


# ============================================
# ENDPOINTS
# ============================================
@app.get("/")
def root():
    return {"service": "ffmpeg", "version": "0.9.0", "status": "running"}


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
async def cut_scenes_endpoint(
    file: UploadFile = File(...),
    scenes: str = Form(...),
    padding: float = Form(0.05),
    fade: float = Form(0.03),
    auth: str = Depends(verify_token)
):
    """Cut multiple scenes from combined audio at exact timestamps."""
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
    Smart context-aware audio optimization.
    
    Targets:
      - Soft target: 3.0s (ideal)
      - Hard cap:    4.0s (absolute, never exceeded)
    
    Logic (v0.9):
      1. Analyze audio (volume, silence, speech density, style)
      2. Edge silence trim (leading + trailing silence always trimmed)
      3. Mid-speech silence trim (style-aware caps)
      4. Atempo (style-scaled; 2% safety margin for ffmpeg approximation)
    
    Form params:
      file:   MP3 file to optimize (required)
    
    Edge silence trim (v0.8.1):
      - Leading silence (pause within 50ms of start): trimmed to 50ms keep
      - Trailing silence (pause within 50ms of end): trimmed to 50ms keep
      - Mid-speech pauses: use style-based rules (preserves breath rhythm)
    
    Atempo caps:
      - Fast speaker:   1.08 (gentle, preserves dense delivery)
      - Normal speaker: 1.12
      - Slow speaker:   1.15
      - Hard cap override: up to 1.15 (or 1.20 emergency) for hitting 4s
    
    Note: v0.9 removed pad_to_duration feature. For post-processing videos,
          use /video/smart-cut + /video/speedup or /video/optimize.
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


# ============================================
# VIDEO HELPERS (v0.9)
# ============================================
def extract_audio_from_video(video_path: Path, audio_path: Path) -> tuple:
    """Extract audio track from video as MP3 for analysis."""
    return run_ffmpeg([
        "-i", str(video_path),
        "-vn",
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(audio_path)
    ])


def cut_video_segment(input_path: Path, output_path: Path, start: float, end: float) -> tuple:
    """Cut video+audio to [start, end] range. Re-encodes for accurate cuts."""
    duration = end - start
    return run_ffmpeg([
        "-accurate_seek",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path)
    ])


def speedup_video(input_path: Path, output_path: Path, speed: float) -> tuple:
    """
    Speed up video+audio with constant pitch.
    setpts=PTS/speed for video, atempo=speed for audio.
    """
    if speed <= 1.001:
        # No meaningful speedup - just copy
        shutil.copy(input_path, output_path)
        return 0, ""
    
    # atempo filter supports 0.5-2.0 range, chain if needed
    atempo_chain = f"atempo={speed:.4f}"
    if speed > 2.0:
        atempo_chain = "atempo=2.0,atempo=" + f"{(speed/2.0):.4f}"
    
    return run_ffmpeg([
        "-i", str(input_path),
        "-filter_complex", f"[0:v]setpts=PTS/{speed:.4f}[v];[0:a]{atempo_chain}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path)
    ])


# ============================================
# UGC PROFILES (v0.9)
# ============================================
UGC_PROFILES = {
    "default": {
        "target_final_duration": 3.0,
        "target_density": 0.85,
        "max_pause_ms": 250,
        "max_speedup": 1.15,
    },
}


def decide_video_optimization(analysis: dict, duration: float, profile: dict) -> dict:
    """
    Decide how to optimize the video based on audio analysis and UGC profile.
    
    Returns:
        dict with decisions:
          - trim_start, trim_end (video cut boundaries)
          - speedup (1.0 if not needed)
          - classification (too_slow/good/too_fast)
          - reasoning
    """
    silences = analysis["pauses"]["all_pauses"]
    
    # Identify edge silences (leading/trailing dead air)
    leading_silence = None
    trailing_silence = None
    
    if silences:
        if silences[0]["start"] <= EDGE_TOLERANCE and silences[0]["duration_ms"] > EDGE_SILENCE_MIN_MS:
            leading_silence = silences[0]
        if silences[-1]["end"] >= duration - EDGE_TOLERANCE and silences[-1]["duration_ms"] > EDGE_SILENCE_MIN_MS:
            if silences[-1] is not leading_silence:
                trailing_silence = silences[-1]
    
    # Calculate cut boundaries (keep 50ms buffer on each edge)
    keep_seconds = EDGE_SILENCE_KEEP_MS / 1000
    if leading_silence:
        trim_start = max(0, leading_silence["end"] - keep_seconds)
    else:
        trim_start = 0.0
    
    if trailing_silence:
        trim_end = min(duration, trailing_silence["start"] + keep_seconds)
    else:
        trim_end = duration
    
    trimmed_duration = trim_end - trim_start
    
    # Recalculate density AFTER trimming (only count mid-speech silence)
    mid_silence_ms = 0
    for s in silences:
        if s is leading_silence or s is trailing_silence:
            continue
        mid_silence_ms += s["duration_ms"]
    
    speech_time = trimmed_duration - (mid_silence_ms / 1000)
    trimmed_density = speech_time / trimmed_duration if trimmed_duration > 0 else 1.0
    
    # Calculate onset rate (speech onsets per second)
    mid_silences = [s for s in silences if s is not leading_silence and s is not trailing_silence]
    onset_rate = (len(mid_silences) + 1) / trimmed_duration if trimmed_duration > 0 else 0
    
    # Pace classification
    avg_mid_pause_ms = (mid_silence_ms / len(mid_silences)) if mid_silences else 0
    
    # Score against profile
    target_density = profile["target_density"]
    density_ratio = trimmed_density / target_density
    
    target_duration = profile["target_final_duration"]
    duration_ratio = trimmed_duration / target_duration
    
    # Classify delivery
    if trimmed_density < target_density * 0.90 or duration_ratio > 1.15:
        classification = "too_slow"
    elif duration_ratio < 0.85 and trimmed_density > target_density * 1.05:
        classification = "too_fast"
    else:
        classification = "good"
    
    # Calculate speedup
    speedup = 1.0
    speedup_reason = "already in target range"
    
    if classification == "too_slow" and trimmed_duration > target_duration:
        ideal_speedup = trimmed_duration / target_duration
        speedup = min(ideal_speedup, profile["max_speedup"])
        speedup = round(speedup, 3)
        speedup_reason = f"speeding up from {trimmed_duration:.2f}s toward {target_duration:.2f}s target"
    elif classification == "too_slow":
        # Slow but already short enough - minimal speedup
        speedup = 1.0
        speedup_reason = "slow pace but duration already acceptable"
    
    final_duration = trimmed_duration / speedup if speedup > 1.0 else trimmed_duration
    
    return {
        "trim_start": round(trim_start, 3),
        "trim_end": round(trim_end, 3),
        "trimmed_duration": round(trimmed_duration, 3),
        "leading_silence_ms": leading_silence["duration_ms"] if leading_silence else 0,
        "trailing_silence_ms": trailing_silence["duration_ms"] if trailing_silence else 0,
        "mid_silence_ms": round(mid_silence_ms),
        "avg_mid_pause_ms": round(avg_mid_pause_ms),
        "trimmed_density": round(trimmed_density, 3),
        "onset_rate": round(onset_rate, 2),
        "classification": classification,
        "speedup": speedup,
        "speedup_reason": speedup_reason,
        "estimated_final_duration": round(final_duration, 3),
    }


# ============================================
# VIDEO ENDPOINTS (v0.9)
# ============================================
@app.post("/video/analyze")
async def video_analyze_endpoint(
    file: UploadFile = File(...),
    auth: str = Depends(verify_token)
):
    """
    Extract audio from video and run full analysis.
    Useful for understanding Seedance output before deciding trim/speedup.
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())[:8]
    video_path = WORK_DIR / f"{job_id}_input.mp4"
    audio_path = WORK_DIR / f"{job_id}_extracted.mp3"
    
    try:
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        video_duration = get_duration(video_path)
        
        ret, err = extract_audio_from_video(video_path, audio_path)
        if ret != 0:
            return {"status": "error", "error": f"audio extraction failed: {err[:200]}"}
        
        analysis = analyze_audio(audio_path)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "status": "ok",
            "video_duration": round(video_duration, 3),
            "audio_analysis": analysis,
            "processing_time_ms": elapsed_ms
        }
    
    finally:
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


@app.post("/video/smart-cut")
async def video_smart_cut_endpoint(
    file: UploadFile = File(...),
    buffer_ms: int = Form(50),
    auth: str = Depends(verify_token)
):
    """
    Auto-trim leading and trailing silence from video.
    Preserves all mid-speech content and pauses.
    
    Uses same silence detection as /audio/optimize edge trim.
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())[:8]
    video_path = WORK_DIR / f"{job_id}_input.mp4"
    audio_path = WORK_DIR / f"{job_id}_extracted.mp3"
    output_path = WORK_DIR / f"{job_id}_cut.mp4"
    
    try:
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        video_duration = get_duration(video_path)
        
        # Extract audio and analyze
        ret, err = extract_audio_from_video(video_path, audio_path)
        if ret != 0:
            return {"status": "error", "error": f"audio extraction failed: {err[:200]}"}
        
        analysis = analyze_audio(audio_path)
        audio_path.unlink(missing_ok=True)
        
        silences = analysis["pauses"]["all_pauses"]
        
        # Detect edge silences
        leading_silence = None
        trailing_silence = None
        
        if silences:
            if silences[0]["start"] <= EDGE_TOLERANCE and silences[0]["duration_ms"] > EDGE_SILENCE_MIN_MS:
                leading_silence = silences[0]
            if silences[-1]["end"] >= video_duration - EDGE_TOLERANCE and silences[-1]["duration_ms"] > EDGE_SILENCE_MIN_MS:
                if silences[-1] is not leading_silence:
                    trailing_silence = silences[-1]
        
        # Calculate cut boundaries
        buffer_s = buffer_ms / 1000
        trim_start = max(0, leading_silence["end"] - buffer_s) if leading_silence else 0.0
        trim_end = min(video_duration, trailing_silence["start"] + buffer_s) if trailing_silence else video_duration
        
        # If no edge silence detected, just pass through
        if trim_start <= 0.01 and trim_end >= video_duration - 0.01:
            shutil.copy(video_path, output_path)
            final_duration = video_duration
            action = "passthrough"
        else:
            ret, err = cut_video_segment(video_path, output_path, trim_start, trim_end)
            if ret != 0:
                return {"status": "error", "error": f"cut failed: {err[:200]}"}
            final_duration = get_duration(output_path)
            action = "cut"
        
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        output_path.unlink(missing_ok=True)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "status": "ok",
            "action": action,
            "original_duration": round(video_duration, 3),
            "trim_start": round(trim_start, 3),
            "trim_end": round(trim_end, 3),
            "final_duration": round(final_duration, 3),
            "leading_silence_trimmed_ms": leading_silence["duration_ms"] - buffer_ms if leading_silence else 0,
            "trailing_silence_trimmed_ms": trailing_silence["duration_ms"] - buffer_ms if trailing_silence else 0,
            "video_base64": video_b64,
            "size_bytes": len(video_bytes),
            "processing_time_ms": elapsed_ms
        }
    
    finally:
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


@app.post("/video/speedup")
async def video_speedup_endpoint(
    file: UploadFile = File(...),
    speed: float = Form(1.15),
    auth: str = Depends(verify_token)
):
    """
    Speed up video + audio with constant pitch.
    Useful for tightening slow-paced Seedance output.
    
    Speed range: 0.5 - 2.0 (recommended: 1.0 - 1.15 for UGC)
    """
    if speed < 0.5 or speed > 2.0:
        raise HTTPException(400, "speed must be between 0.5 and 2.0")
    
    start_time = time.time()
    job_id = str(uuid.uuid4())[:8]
    video_path = WORK_DIR / f"{job_id}_input.mp4"
    output_path = WORK_DIR / f"{job_id}_speedup.mp4"
    
    try:
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        original_duration = get_duration(video_path)
        
        ret, err = speedup_video(video_path, output_path, speed)
        if ret != 0:
            return {"status": "error", "error": f"speedup failed: {err[:200]}"}
        
        final_duration = get_duration(output_path)
        
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        output_path.unlink(missing_ok=True)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "status": "ok",
            "original_duration": round(original_duration, 3),
            "speed_applied": round(speed, 3),
            "final_duration": round(final_duration, 3),
            "video_base64": video_b64,
            "size_bytes": len(video_bytes),
            "processing_time_ms": elapsed_ms
        }
    
    finally:
        video_path.unlink(missing_ok=True)


@app.post("/video/optimize")
async def video_optimize_endpoint(
    file: UploadFile = File(...),
    target_duration: Optional[float] = Form(None),
    max_speedup: Optional[float] = Form(None),
    apply_trim: bool = Form(True),
    apply_speedup: bool = Form(True),
    auth: str = Depends(verify_token)
):
    """
    Combined video optimization: analyze + smart-cut + auto-speedup.
    
    Logic (v0.9):
      1. Extract audio, run full analysis
      2. Calculate optimal trim boundaries + speedup based on UGC profile
      3. Apply: trim edges, then speedup if still too slow
      4. Return optimized video + decision details
    
    Default profile:
      - target_final_duration: 3.0s
      - target_density: 0.85
      - max_speedup: 1.15x
      - max_pause_ms: 250
    
    Form params:
      file:              Video file (MP4)
      target_duration:   Optional override for target duration
      max_speedup:       Optional override for max speedup cap
      apply_trim:        If True, trim leading/trailing silence (default: True)
      apply_speedup:     If True, apply speedup if classified too_slow (default: True)
    """
    profile_settings = dict(UGC_PROFILES["default"])
    if target_duration is not None:
        profile_settings["target_final_duration"] = target_duration
    if max_speedup is not None:
        profile_settings["max_speedup"] = max_speedup
    
    start_time = time.time()
    job_id = str(uuid.uuid4())[:8]
    video_path = WORK_DIR / f"{job_id}_input.mp4"
    audio_path = WORK_DIR / f"{job_id}_extracted.mp3"
    cut_path = WORK_DIR / f"{job_id}_cut.mp4"
    final_path = WORK_DIR / f"{job_id}_final.mp4"
    
    try:
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        video_duration = get_duration(video_path)
        
        # Step 1: Extract audio and analyze
        ret, err = extract_audio_from_video(video_path, audio_path)
        if ret != 0:
            return {"status": "error", "error": f"audio extraction failed: {err[:200]}"}
        
        analysis = analyze_audio(audio_path)
        audio_path.unlink(missing_ok=True)
        
        # Step 2: Decide optimizations based on profile
        decision = decide_video_optimization(analysis, video_duration, profile_settings)
        
        # Step 3: Apply trim if enabled and beneficial
        current_path = video_path
        actual_trim_applied = False
        
        if apply_trim and (decision["trim_start"] > 0.01 or decision["trim_end"] < video_duration - 0.01):
            ret, err = cut_video_segment(
                video_path, cut_path,
                decision["trim_start"], decision["trim_end"]
            )
            if ret != 0:
                return {"status": "error", "error": f"cut failed: {err[:200]}"}
            current_path = cut_path
            actual_trim_applied = True
        
        # Step 4: Apply speedup if enabled and needed
        actual_speedup = 1.0
        if apply_speedup and decision["speedup"] > 1.01:
            ret, err = speedup_video(current_path, final_path, decision["speedup"])
            if ret != 0:
                return {"status": "error", "error": f"speedup failed: {err[:200]}"}
            actual_speedup = decision["speedup"]
            if current_path == cut_path:
                cut_path.unlink(missing_ok=True)
            current_path = final_path
        
        # If no operation applied, use original
        if current_path == video_path:
            shutil.copy(video_path, final_path)
            current_path = final_path
        elif current_path != final_path:
            # Move cut result to final
            current_path.rename(final_path)
            current_path = final_path
        
        final_duration = get_duration(final_path)
        
        with open(final_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        final_path.unlink(missing_ok=True)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Determine action
        if actual_trim_applied and actual_speedup > 1.0:
            action = "trim_and_speedup"
        elif actual_trim_applied:
            action = "trim_only"
        elif actual_speedup > 1.0:
            action = "speedup_only"
        else:
            action = "passthrough"
        
        return {
            "status": "ok",
            "action": action,
            "profile_settings": profile_settings,
            "original_duration": round(video_duration, 3),
            "audio_analysis": {
                "style": analysis["speech"]["style"],
                "density": analysis["speech"]["speech_density"],
                "total_silence_ms": analysis["pauses"]["total_ms"],
                "pause_count": analysis["pauses"]["count"],
            },
            "decision": decision,
            "applied": {
                "trim_applied": actual_trim_applied,
                "speedup_applied": round(actual_speedup, 3),
            },
            "final_duration": round(final_duration, 3),
            "video_base64": video_b64,
            "size_bytes": len(video_bytes),
            "processing_time_ms": elapsed_ms
        }
    
    finally:
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
        cut_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
