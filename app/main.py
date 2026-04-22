"""
FFmpeg HTTP Service v0.8.1
Smart audio cutting + context-aware optimization + edge silence trim + symmetric padding for UGC ad pipeline.

Endpoints:
  GET  /                    - Service info
  GET  /health              - Health check + ffmpeg version
  GET  /auth-test           - Auth verification
  POST /audio/cut-scenes    - Cut multiple scenes from combined audio
  POST /audio/optimize      - Smart optimization (target 3s, hard cap 4s, optional symmetric padding)

Changelog v0.8.1:
  - NEW: Auto-trim leading silence (pauses at very start of audio, >50ms)
  - NEW: Auto-trim trailing silence (pauses at very end of audio, >50ms)
  - Edge silences are ALWAYS trimmed regardless of style rules
  - Mid-speech pauses still use style-based rules (preserves natural breath)
  - Significantly improves silence savings for files with TTS lead-in/lead-out silence
  - 2% safety margin on atempo calculation (avoid ffmpeg approximation overshoot)

Changelog v0.8:
  - NEW: Symmetric padding feature for Seedance video generation
  - Optional form param `pad_to_duration` (float, 0.5-15.0s) on /audio/optimize
  - When set, pads audio symmetrically (front + back silence) to reach exact target duration
  - Example: content 2.6s + pad_to_duration=4.0 -> 0.7s silence + 2.6s content + 0.7s silence = 4.0s
  - Helps Seedance render full video duration with complete voice delivery
  - No-op if audio is already >= target duration

Changelog v0.7:
  - Removed /audio/analyze endpoint (analysis bundled inside /audio/optimize)
  - Style-scaled atempo caps: Fast 1.08, Normal 1.12, Slow 1.15
  - Atempo applied to all styles between soft target and hard cap
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

app = FastAPI(title="FFmpeg Service", version="0.8.1")

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

# Padding limits (v0.8)
PAD_TO_DURATION_MIN = 0.5
PAD_TO_DURATION_MAX = 15.0

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


def apply_symmetric_padding(input_path: Path, output_path: Path, target_duration: float) -> tuple:
    """
    Pad audio symmetrically with silence to reach exact target duration.
    Example: content 2.6s, target 4.0s -> 0.7s silence + 2.6s content + 0.7s silence = 4.0s
    """
    current_duration = get_duration(input_path)
    
    if current_duration >= target_duration:
        shutil.copy(input_path, output_path)
        return 0, ""
    
    total_padding = target_duration - current_duration
    half_padding = total_padding / 2
    pad_front_ms = int(half_padding * 1000)
    pad_back_seconds = half_padding
    
    return run_ffmpeg([
        "-i", str(input_path),
        "-af", f"adelay={pad_front_ms}|{pad_front_ms},apad=pad_dur={pad_back_seconds:.3f}",
        "-t", f"{target_duration:.3f}",
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(output_path)
    ])


def optimize_audio(input_path: Path, job_id: str, pad_to_duration: Optional[float] = None) -> dict:
    """Full optimization pipeline: analyze + context-aware execution + optional symmetric padding."""
    # STEP 1: Analyze
    analysis = analyze_audio(input_path)
    duration = analysis["duration"]
    style = analysis["speech"]["style"]
    silences = analysis["pauses"]["all_pauses"]
    
    padding_info = {
        "requested": pad_to_duration is not None,
        "target_duration": pad_to_duration,
        "applied": False,
        "pad_front_ms": 0,
        "pad_back_ms": 0
    }
    
    # STEP 2: Short file passthrough path
    if duration <= SOFT_TARGET:
        passthrough_path = WORK_DIR / f"{job_id}_passthrough.mp3"
        shutil.copy(input_path, passthrough_path)
        
        if pad_to_duration is not None and duration < pad_to_duration:
            padded_path = WORK_DIR / f"{job_id}_padded.mp3"
            ret, err = apply_symmetric_padding(passthrough_path, padded_path, pad_to_duration)
            passthrough_path.unlink(missing_ok=True)
            
            if ret != 0:
                padded_path.unlink(missing_ok=True)
                return {"status": "error", "error": f"padding failed: {err[:200]}"}
            
            final_path = padded_path
            total_padding = pad_to_duration - duration
            half_padding = total_padding / 2
            padding_info["applied"] = True
            padding_info["pad_front_ms"] = round(half_padding * 1000)
            padding_info["pad_back_ms"] = round(half_padding * 1000)
            final_duration = get_duration(final_path)
        else:
            final_path = passthrough_path
            final_duration = duration
        
        with open(final_path, "rb") as f:
            audio_bytes = f.read()
        final_path.unlink(missing_ok=True)
        
        return {
            "status": "ok",
            "action": "passthrough" + ("_padded" if padding_info["applied"] else ""),
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
            "padding": padding_info,
            "original_duration": round(duration, 3),
            "content_duration": round(duration, 3),
            "final_duration": round(final_duration, 3),
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
    
    # STEP 5: Apply symmetric padding (v0.8)
    if pad_to_duration is not None and content_duration < pad_to_duration:
        padded_path = WORK_DIR / f"{job_id}_padded.mp3"
        ret, err = apply_symmetric_padding(optimized_path, padded_path, pad_to_duration)
        optimized_path.unlink(missing_ok=True)
        
        if ret != 0:
            padded_path.unlink(missing_ok=True)
            return {"status": "error", "error": f"padding failed: {err[:200]}"}
        
        total_padding = pad_to_duration - content_duration
        half_padding = total_padding / 2
        padding_info["applied"] = True
        padding_info["pad_front_ms"] = round(half_padding * 1000)
        padding_info["pad_back_ms"] = round(half_padding * 1000)
        
        final_path = padded_path
    else:
        final_path = optimized_path
    
    final_duration = get_duration(final_path)
    
    with open(final_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    final_path.unlink(missing_ok=True)
    
    action = "optimized"
    if padding_info["applied"]:
        action += "_padded"
    
    return {
        "status": "ok",
        "action": action,
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
        "padding": padding_info,
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
    return {"service": "ffmpeg", "version": "0.8.1", "status": "running"}


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
    pad_to_duration: Optional[float] = Form(None),
    auth: str = Depends(verify_token)
):
    """
    Smart context-aware audio optimization with optional symmetric padding.
    
    Targets:
      - Soft target: 3.0s (ideal)
      - Hard cap:    4.0s (absolute, never exceeded)
    
    Logic (v0.8.1):
      1. Analyze audio (volume, silence, speech density, style)
      2. Edge silence trim (leading + trailing silence always trimmed)
      3. Mid-speech silence trim (style-aware caps)
      4. Atempo (style-scaled; 2% safety margin for ffmpeg approximation)
      5. Optional: symmetric padding to reach exact target duration
    
    Form params:
      file:               MP3 file to optimize (required)
      pad_to_duration:    Optional. Float, 0.5-15.0s. Pads final audio symmetrically
                          to reach exact duration. Helps Seedance render full video
                          duration with complete voice delivery.
    
    Edge silence trim (v0.8.1):
      - Leading silence (pause within 50ms of start): trimmed to 50ms keep
      - Trailing silence (pause within 50ms of end): trimmed to 50ms keep
      - Mid-speech pauses: use style-based rules (preserves breath rhythm)
    
    Atempo caps:
      - Fast speaker:   1.08 (gentle, preserves dense delivery)
      - Normal speaker: 1.12
      - Slow speaker:   1.15
      - Hard cap override: up to 1.15 (or 1.20 emergency) for hitting 4s
    """
    start_time = time.time()
    
    if pad_to_duration is not None:
        if pad_to_duration < PAD_TO_DURATION_MIN or pad_to_duration > PAD_TO_DURATION_MAX:
            raise HTTPException(
                400,
                f"pad_to_duration must be between {PAD_TO_DURATION_MIN} and {PAD_TO_DURATION_MAX} seconds"
            )
    
    job_id = str(uuid.uuid4())[:8]
    input_path = WORK_DIR / f"{job_id}_input.mp3"

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        result = optimize_audio(input_path, job_id, pad_to_duration=pad_to_duration)
        elapsed_ms = int((time.time() - start_time) * 1000)
        result["processing_time_ms"] = elapsed_ms
        
        return result

    finally:
        input_path.unlink(missing_ok=True)
