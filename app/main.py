"""
FFmpeg HTTP Service v0.11.0
Smart audio cutting + context-aware optimization + video optimization for UGC ad pipeline.

Endpoints:
  GET  /                    - Service info
  GET  /health              - Health check + ffmpeg version
  GET  /auth-test           - Auth verification
  POST /audio/snap-scenes   - Post-speech aware snap (v0.11.0)
  POST /audio/cut-scenes    - Cut multiple scenes from combined audio
  POST /audio/optimize      - Smart optimization (target 3s, hard cap 4s, edge silence trim)
  POST /video/analyze       - Analyze video's audio track
  POST /video/smart-cut     - Auto-trim leading/trailing silence from video
  POST /video/speedup       - Speed up video + audio with constant pitch
  POST /video/merge-audio   - Merge silent video with audio track (v0.9.6)
  POST /video/optimize      - Combined: smart-cut + auto-speedup + audio fade-in/out
  POST /video/preset        - Lightroom-style color grading (v0.9.12)

Changelog v0.11.0:
  - FIXED: find_gap_quietest no longer cuts off final plosives (d, t, p, b, k, g)
  - Old behavior: searched ±buffer_ms around the ELL gap, picked absolute RMS
    minimum. This sometimes found the stop-closure stille INSIDE a plosive
    (e.g. inside the 'd' of "good") and cut there, removing the burst.
  - New behavior:
    * Asymmetric search window — 30ms back, 250ms forward (default)
    * Post-speech detection: dynamically computes speech-floor threshold
      from window's 75th percentile RMS, walks forward to find LAST chunk
      with speech-level energy, adds 60ms grace for plosive bursts/fricative
      tails, THEN searches for quietest point AFTER that
    * Finer chunk granularity (10ms vs old 20ms)
  - Backward-compatible: same function signature, same return dict keys
    (plus two new debug fields: speech_floor_db, last_speech_idx)
  - Legacy buffer_ms=100 now maps to forward_buffer_ms=250 (was symmetric)
  - Snap-scenes endpoint behavior improves automatically; no caller changes

Changelog v0.10.1:
  - CHANGED: Trailing trim buffer 25ms → 50ms (symmetric with leading)
  - CHANGED: /video/optimize now applies audio-only fade-in/out (no padding,
    no video fade). Video remains stream-copied (lossless + fast).
  - Audio fade happens mostly over the trim buffer zone (where audio is mostly
    silent), with minimal touch on speech edges.
  - When scenes are concatenated downstream, audio bridges smoothly.

Changelog v0.10.0:
  - REMOVED: /video/concat endpoint (replaced by external service for assembly)
  - Per-scene processing only — clean main.py
  - Removed: build_concat_filter_chain helper, ConcatRequest model

Changelog v0.9.12:
  - NEW: /video/preset endpoint — Lightroom-style color grading
  - 8 settings on Lightroom scale (-100 to +100):
    temp, tint, saturation, exposure, contrast, highlight, shadow, fade
  - Default preset matches "modern UGC faded" aesthetic:
    temp=-3, tint=+2, sat=-6, exp=-3, contrast=+12, hl=-35, sh=+18, fade=+6
  - Audio stream copied unchanged (no re-encode for audio)
  - Use as final step: ... → /video/optimize → /video/preset → upload
  - All params optional — call with no body for default UGC preset

Changelog v0.9.11:
  - NEW DEFAULT MODE: gap_quietest
  - Instead of scanning ±window around EACH timestamp independently (v0.9.10),
    now scan the GAP BETWEEN adjacent scenes and find ONE shared cut point
  - Advantages:
    * GUARANTEED no overlaps (both scenes use same snap point)
    * Window is defined by actual gap → searches right zone
    * Cannot bleed into wrong scene's audio
    * Naturally adapts to any gap size
  - Parameters:
    * buffer_ms (default 100): extra margin on each side of the ELL gap
    * chunk_ms (default 20): RMS measurement resolution
  - Old modes still available via `mode=rms_minimum` or `mode=silence`

Changelog v0.9.10:
  - IMPROVED: snap-scenes uses RMS-MINIMUM (replaced by gap_quietest in v0.9.11)

Changelog v0.9.9:
  - NEW: /audio/snap-scenes endpoint — hybrid timestamp correction
  - Takes ElevenLabs character-level timestamps as HINT, then acoustically
    snaps each scene boundary to the nearest real silence midpoint
  - Use as pre-processor before /audio/cut-scenes for max precision
  - Flow: TTS → snap-scenes → cut-scenes → optimize
  - Fallback: if no silence found in window, keeps original timestamp
  - Prevents overlaps by coordinating adjacent scene boundaries

Changelog v0.9.8:
  - Reverted default padding in /audio/cut-scenes back to 0.05
  - Reason: Root cause of cut-off fricatives was upstream (combined VO had no
    sentence breaks between scenes). Fixed in voiceover_assembly_v1.3.js by
    joining scenes with ". " instead of " " — natural pauses between scenes
    now give ElevenLabs proper word boundaries.
  - With proper sentence breaks upstream, minimal padding (0.05) is sufficient.

Changelog v0.9.7 (superseded):
  - Increased default padding to 0.2 — turned out to be patching a symptom.
  - Real fix is upstream in VO assembly (see v0.9.8).

Changelog v0.9.6:
  - NEW: /video/merge-audio endpoint for BROLL post-production
  - Use case: Seedance generates silent BROLL (generate_audio: false)
    then this endpoint attaches the ElevenLabs VO as audio track
  - Flow: Seedance silent video → /video/merge-audio → /video/optimize
  - /video/optimize UNCHANGED

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
import urllib.request
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI(title="FFmpeg Service", version="0.11.0")

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
EDGE_SILENCE_MIN_MS = 50              # trim edges only if they exceed this
EDGE_SILENCE_KEEP_LEADING_MS = 50     # buffer at start (natural fade-in)
EDGE_SILENCE_KEEP_TRAILING_MS = 50    # buffer at end (symmetric with leading)
EDGE_SILENCE_KEEP_MS = 50             # legacy fallback (kept for compatibility)
EDGE_TOLERANCE = 0.05                 # tolerance for detecting edge position (50ms)

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials


# ============================================
# BASIC HELPERS
# ============================================
def run_ffmpeg(args: list, timeout: int = 60) -> tuple:
    """Run ffmpeg command, return (returncode, stderr)."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stderr
    except subprocess.TimeoutExpired:
        return -1, f"ffmpeg timeout after {timeout}s"


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


def analyze_rms_windows(input_path: Path, start: float, end: float, window_ms: int = 100) -> list:
    """
    Analyze RMS energy by slicing audio into windows and measuring each.
    Returns list of (window_start_s, rms_db) tuples.
    
    Uses slice-by-slice ffmpeg calls which gives clean per-window RMS
    (the cumulative astats approach doesn't work for this use case).
    """
    if end - start <= 0.05:
        return []
    
    window_s = window_ms / 1000
    num_windows = int((end - start) / window_s)
    
    windows = []
    for i in range(num_windows):
        win_start = start + i * window_s
        
        cmd = [
            "ffmpeg", "-hide_banner",
            "-ss", f"{win_start:.3f}",
            "-i", str(input_path),
            "-t", f"{window_s:.3f}",
            "-af", "astats=metadata=1,ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        # Get the LAST RMS value (final summary after this window)
        rms_values = re.findall(r"lavfi\.astats\.Overall\.RMS_level=(-?[\d.]+|-?inf)", result.stderr)
        
        if rms_values:
            rms_str = rms_values[-1]
            if rms_str == "-inf":
                rms_db = -100.0
            else:
                rms_db = float(rms_str)
            windows.append((round(win_start, 3), round(rms_db, 2)))
    
    return windows


def detect_trailing_breath(input_path: Path, analysis: dict, scan_duration: float = 1.0) -> float:
    """
    Detect trailing breath/puste that silencedetect missed.
    
    Strategy:
      1. Calculate speech baseline from full-file mean RMS
      2. Slice last N seconds into 100ms windows, measure each
      3. Walk backwards to find last window with speech-level energy
      4. Return where real speech ends (+ grace period for fricatives)
    
    Returns: estimated real speech end time (seconds from start of file)
             If no breath detected, returns file duration unchanged.
    """
    duration = analysis["duration"]
    mean_db = analysis["volume"]["mean_db"]
    
    if mean_db is None or duration < scan_duration:
        return duration  # can't analyze
    
    # Speech floor: RMS threshold for real speech vs breath
    # Set at mean - 15dB (was -10dB - too aggressive for fricatives like 's', 'f', 'sh')
    # Fricatives are quieter than vowels but still real speech
    speech_floor_db = mean_db - 15.0
    
    # Scan the last `scan_duration` seconds in 100ms windows
    scan_start = max(0, duration - scan_duration)
    windows = analyze_rms_windows(input_path, scan_start, duration, window_ms=100)
    
    if not windows:
        return duration
    
    # Walk backwards: find last window with speech-level energy
    last_speech_time = duration
    found_speech = False
    
    for window_time, rms_db in reversed(windows):
        if rms_db >= speech_floor_db:
            # This window has speech-level energy
            last_speech_time = window_time + 0.1  # add window length (100ms)
            found_speech = True
            break
    
    if not found_speech:
        return duration
    
    # Add fricative grace period - consonants like 's','f','sh' decay slowly
    # and their RMS tail can extend 80-150ms beyond the detected end
    FRICATIVE_GRACE_MS = 120
    last_speech_time += FRICATIVE_GRACE_MS / 1000
    last_speech_time = min(last_speech_time, duration)
    
    # Only trim if breath tail is significant
    breath_duration = duration - last_speech_time
    if breath_duration < 0.08:  # less than 80ms - not worth it
        return duration
    
    return round(last_speech_time, 3)


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
    
    # Handle leading silence: start speech after leading silence (keep 50ms buffer)
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
    return {"service": "ffmpeg", "version": "0.11.0", "status": "running"}


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


# ============================================
# v0.9.9 — SNAP SCENE TIMESTAMPS TO SILENCE
# Hybrid approach: use ElevenLabs timestamps as hint,
# then acoustic-snap to nearest real silence for precision.
# ============================================
def find_nearest_silence(
    target_time: float,
    silences: list,
    window_s: float = 0.5,
    min_silence_ms: int = 150
) -> Optional[dict]:
    """
    Find the silence closest to target_time within window, with min duration.
    
    Args:
        target_time: timestamp we want to snap to (from ElevenLabs alignment)
        silences: list of detected silences [{start, end, duration, duration_ms}, ...]
        window_s: search window (+/- seconds around target)
        min_silence_ms: minimum silence duration to count as valid
    
    Returns:
        Best matching silence dict, or None if no match
    """
    window_start = target_time - window_s
    window_end = target_time + window_s
    
    candidates = []
    for s in silences:
        # Check min duration
        if s.get("duration_ms", int(s["duration"] * 1000)) < min_silence_ms:
            continue
        
        # Check if silence overlaps the search window
        # (silence must have ANY part within window)
        if s["end"] < window_start or s["start"] > window_end:
            continue
        
        # Calculate midpoint and distance to target
        midpoint = (s["start"] + s["end"]) / 2
        distance = abs(midpoint - target_time)
        candidates.append((distance, s, midpoint))
    
    if not candidates:
        return None
    
    # Sort by distance, pick closest
    candidates.sort(key=lambda x: x[0])
    return {
        "silence": candidates[0][1],
        "midpoint": candidates[0][2],
        "distance_ms": round(candidates[0][0] * 1000),
    }


def find_rms_minimum(
    input_path: Path,
    target_time: float,
    audio_duration: float,
    window_s: float = 0.3,
    chunk_ms: int = 20
) -> dict:
    """
    Find the quietest point in a window around target_time.
    
    This is MORE ROBUST than silence detection because it always finds
    a minimum (even when there's no "true silence"). It uses the
    ElevenLabs timestamp as an anchor and finds the local RMS valley
    in the search window.
    
    Args:
        input_path: audio file
        target_time: anchor timestamp (from ElevenLabs alignment)
        audio_duration: total audio duration (for clamping)
        window_s: search window (+/- seconds around target)
        chunk_ms: size of each RMS measurement chunk (smaller = more precise)
    
    Returns:
        dict with:
          - quietest_time: timestamp of the quietest point
          - quietest_rms_db: RMS in dB at that point
          - window_start, window_end: the search range used
          - chunks_measured: how many chunks were analyzed
          - distance_from_target_ms: how far from ElevenLabs original
    """
    window_start = max(0, target_time - window_s)
    window_end = min(audio_duration, target_time + window_s)
    
    if window_end - window_start < 0.05:
        # Window too small — fallback to target
        return {
            "quietest_time": target_time,
            "quietest_rms_db": None,
            "window_start": window_start,
            "window_end": window_end,
            "chunks_measured": 0,
            "distance_from_target_ms": 0,
            "fallback_reason": "window_too_small",
        }
    
    # Measure RMS in small chunks across the window
    windows = analyze_rms_windows(
        input_path, window_start, window_end, window_ms=chunk_ms
    )
    
    if not windows:
        return {
            "quietest_time": target_time,
            "quietest_rms_db": None,
            "window_start": window_start,
            "window_end": window_end,
            "chunks_measured": 0,
            "distance_from_target_ms": 0,
            "fallback_reason": "no_rms_data",
        }
    
    # Find the chunk with the LOWEST RMS (quietest point)
    # windows is list of (time, rms_db) tuples
    quietest = min(windows, key=lambda w: w[1])
    quietest_time = quietest[0] + (chunk_ms / 1000) / 2  # center of chunk
    quietest_rms = quietest[1]
    
    return {
        "quietest_time": round(quietest_time, 3),
        "quietest_rms_db": round(quietest_rms, 2),
        "window_start": round(window_start, 3),
        "window_end": round(window_end, 3),
        "chunks_measured": len(windows),
        "distance_from_target_ms": round(abs(quietest_time - target_time) * 1000),
    }


def find_gap_quietest(
    input_path: Path,
    prev_end: float,
    next_start: float,
    audio_duration: float,
    buffer_ms: int = 100,
    chunk_ms: int = 10,
    back_buffer_ms: Optional[int] = None,
    forward_buffer_ms: Optional[int] = None,
    plosive_grace_ms: int = 60,
) -> dict:
    """
    Find the cut point AFTER the last speech energy in the gap between
    two scenes (v0.11.0 — post-speech aware).

    Asymmetric search window (mostly forward) + post-speech detection
    ensures we never cut into a final plosive (d, t, p, b, k, g) or
    fricative tail (s, sh, f, z).

    OLD BEHAVIOR (v0.9.11–v0.10.1): searched ±buffer_ms around the gap,
    picked absolute RMS minimum. Sometimes found the stop-closure stille
    INSIDE a final plosive (e.g. inside the 'd' of "good") and cut there,
    removing the burst.

    NEW BEHAVIOR (v0.11.0):
      1. Asymmetric window: 30ms back, 250ms forward (default)
      2. Compute dynamic speech-floor from this window's 75th-pct RMS - 6dB
      3. Walk forward through chunks, find LAST chunk above speech floor
      4. Add 60ms grace for plosive bursts and fricative decay tails
      5. AFTER target_idx, find the quietest chunk → that's the cut point

    Args:
        input_path: audio file
        prev_end: end timestamp of previous scene (from ElevenLabs)
        next_start: start timestamp of next scene (from ElevenLabs)
        audio_duration: total audio duration (for clamping)
        buffer_ms: legacy symmetric buffer; if back/forward_buffer_ms are
            None, gets mapped to forward_buffer_ms (back stays at safe 30ms).
            Default 100 → forward becomes 250 (new aggressive default).
        chunk_ms: RMS measurement granularity (default 10ms — finer than
            old default 20ms for better plosive detection)
        back_buffer_ms: explicit back-search safety margin (default 30ms)
        forward_buffer_ms: explicit forward-search range (default 250ms)
        plosive_grace_ms: grace period after detected last speech energy

    Returns:
        dict with:
          - cut_point: the chosen timestamp (shared boundary)
          - cut_rms_db: RMS energy at that point
          - gap_ms: original gap between ElevenLabs timestamps
          - window_start, window_end: actual search range
          - chunks_measured: number of RMS samples taken
          - moved_from_midpoint_ms: distance from naive gap midpoint
          - speech_floor_db: dynamic threshold used (NEW debug field)
          - last_speech_idx: which chunk had last speech energy (NEW debug)
    """
    # Resolve asymmetric buffers from legacy param if not given explicitly
    if back_buffer_ms is None:
        back_buffer_ms = 30                          # safe minimal back-search
    if forward_buffer_ms is None:
        # Map legacy buffer_ms to forward (it was symmetric, forward matters)
        forward_buffer_ms = max(buffer_ms, 100)
        # If caller passed default buffer_ms=100, use new aggressive 250
        if buffer_ms == 100:
            forward_buffer_ms = 250
    
    back_s = back_buffer_ms / 1000
    fwd_s = forward_buffer_ms / 1000
    
    # Asymmetric window: minimal back, aggressive forward
    window_start = max(0, prev_end - back_s)
    window_end = min(audio_duration, next_start + fwd_s)
    
    gap_ms = round((next_start - prev_end) * 1000)
    gap_midpoint = (prev_end + next_start) / 2
    
    # ============================================
    # FALLBACK 1: Window too small to analyze
    # ============================================
    if window_end - window_start < 0.05:
        return {
            "cut_point": round(gap_midpoint, 3),
            "cut_rms_db": None,
            "gap_ms": gap_ms,
            "window_start": round(window_start, 3),
            "window_end": round(window_end, 3),
            "chunks_measured": 0,
            "moved_from_midpoint_ms": 0,
            "speech_floor_db": None,
            "last_speech_idx": None,
            "fallback_reason": "window_too_small",
        }
    
    # Measure RMS in small chunks across the asymmetric window
    windows = analyze_rms_windows(
        input_path, window_start, window_end, window_ms=chunk_ms
    )
    
    # ============================================
    # FALLBACK 2: No RMS data
    # ============================================
    if not windows:
        return {
            "cut_point": round(gap_midpoint, 3),
            "cut_rms_db": None,
            "gap_ms": gap_ms,
            "window_start": round(window_start, 3),
            "window_end": round(window_end, 3),
            "chunks_measured": 0,
            "moved_from_midpoint_ms": 0,
            "speech_floor_db": None,
            "last_speech_idx": None,
            "fallback_reason": "no_rms_data",
        }
    
    # ============================================
    # POST-SPEECH DETECTION (the actual fix)
    # ============================================
    # Step 1: Dynamic speech-floor threshold from this window's data.
    # 75th-pct RMS approximates speech-level energy; -6dB tolerance covers
    # quieter-than-vowel speech like fricatives (s, sh, f).
    rms_values = [w[1] for w in windows]
    sorted_rms = sorted(rms_values)
    pct75_idx = (len(sorted_rms) * 3) // 4
    speech_floor_db = sorted_rms[pct75_idx] - 6.0
    
    # Step 2: Walk forward, find LAST chunk with speech-level energy.
    last_speech_idx = 0
    for i, (_, rms) in enumerate(windows):
        if rms >= speech_floor_db:
            last_speech_idx = i
    
    # Step 3: Add plosive grace — bursts (d,t,p) and fricative tails (s,sh)
    # decay over 60-120ms after main speech energy ends.
    grace_chunks = max(1, plosive_grace_ms // chunk_ms)
    target_idx = min(last_speech_idx + grace_chunks, len(windows) - 1)
    
    # Step 4: AFTER last speech + grace, find the actual quietest chunk.
    # Crucially: never look BEFORE target_idx — can't cut into speech.
    search_window = windows[target_idx:]
    if search_window:
        quietest = min(search_window, key=lambda w: w[1])
    else:
        # Edge case: speech extends to end of window. Best effort fallback.
        quietest = windows[last_speech_idx]
    
    cut_point = quietest[0] + (chunk_ms / 1000) / 2  # center of chunk
    cut_rms = quietest[1]
    
    return {
        "cut_point": round(cut_point, 3),
        "cut_rms_db": round(cut_rms, 2),
        "gap_ms": gap_ms,
        "window_start": round(window_start, 3),
        "window_end": round(window_end, 3),
        "chunks_measured": len(windows),
        "moved_from_midpoint_ms": round(abs(cut_point - gap_midpoint) * 1000),
        "speech_floor_db": round(speech_floor_db, 2),
        "last_speech_idx": last_speech_idx,
    }


@app.post("/audio/snap-scenes")
async def snap_scenes_endpoint(
    file: UploadFile = File(...),
    scenes: str = Form(...),
    buffer_ms: int = Form(100),
    chunk_ms: int = Form(20),
    mode: str = Form("gap_quietest"),
    search_window_ms: int = Form(300),
    min_silence_ms: int = Form(150),
    auth: str = Depends(verify_token)
):
    """
    Snap scene boundaries to the post-speech quiet point between adjacent scenes.
    
    v0.11.0 GAP_QUIETEST MODE (default):
      For each pair of adjacent scenes, find the cut point AFTER the last
      detected speech energy (with grace period for plosive decay).
      Both scene boundaries (prev end + next start) snap to this shared point.
    
    Why post-speech aware (v0.11.0):
      - Final plosives (d, t, p, b, k, g) have a stop-closure stille INSIDE
        the word followed by a burst. Old algorithm sometimes cut at the
        closure, removing the burst.
      - Fricatives (s, sh, f, z) decay slower than vowels and old algorithm
        sometimes treated their tail as silence and cut into them.
      - New algorithm walks forward to find LAST speech-level energy, adds
        60ms grace, THEN looks for the quietest point AFTER that.
    
    Form params:
      file:             Combined audio file (MP3)
      scenes:           JSON array [{part, start, end}, ...]
      mode:             'gap_quietest' (DEFAULT) | 'rms_minimum' | 'silence'
      buffer_ms:        Legacy symmetric buffer (default 100ms) — internally
                        mapped to forward_buffer_ms=250 (back stays 30ms)
      chunk_ms:         RMS measurement granularity (default 20ms; in
                        gap_quietest mode internal default is 10ms for
                        finer plosive detection)
      search_window_ms: Used in 'rms_minimum' and 'silence' modes only
      min_silence_ms:   Used in 'silence' mode only
    
    Returns:
      - Adjusted scenes array (no gaps, no overlaps, cuts in real silence)
      - Per-boundary snap details (cut_point, RMS, speech_floor_db, etc.)
    """
    if mode not in ("gap_quietest", "rms_minimum", "silence"):
        raise HTTPException(400, "mode must be 'gap_quietest', 'rms_minimum', or 'silence'")
    
    try:
        scenes_list = json.loads(scenes)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid scenes JSON: {e}")
    
    if not isinstance(scenes_list, list) or not scenes_list:
        raise HTTPException(400, "scenes must be a non-empty array")
    
    for i, scene in enumerate(scenes_list):
        if "part" not in scene or "start" not in scene or "end" not in scene:
            raise HTTPException(400, f"Scene #{i} missing required fields")
    
    start_time = time.time()
    job_id = str(uuid.uuid4())[:8]
    input_path = WORK_DIR / f"{job_id}_input.mp3"
    
    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        
        audio_duration = get_duration(input_path)
        window_s = search_window_ms / 1000
        
        # Pre-analyze silences only if mode requires it
        silences = []
        silence_threshold_db = None
        if mode == "silence":
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
        
        # Sort scenes by start time (defensive)
        scenes_sorted = sorted(scenes_list, key=lambda s: float(s["start"]))
        
        # ============================================
        # GAP-BASED MODE (v0.11.0 default — post-speech aware)
        # ============================================
        if mode == "gap_quietest":
            adjusted_scenes = []
            snap_metadata = []
            cut_points = []
            
            # Step 1: For each pair of adjacent scenes, find shared cut point
            for i in range(len(scenes_sorted) - 1):
                prev_end = float(scenes_sorted[i]["end"])
                next_start = float(scenes_sorted[i + 1]["start"])
                
                # Safety: if ElevenLabs has overlapping timestamps, swap
                if prev_end > next_start:
                    search_start = next_start
                    search_end = prev_end
                else:
                    search_start = prev_end
                    search_end = next_start
                
                # Use chunk_ms=10 for plosive detection (finer than old 20)
                # unless caller explicitly requested something else via Form
                effective_chunk_ms = 10 if chunk_ms == 20 else chunk_ms
                
                result = find_gap_quietest(
                    input_path,
                    search_start,
                    search_end,
                    audio_duration,
                    buffer_ms=buffer_ms,
                    chunk_ms=effective_chunk_ms,
                )
                cut_points.append(result)
            
            # Step 2: Apply cut points to build adjusted scenes
            for i, scene in enumerate(scenes_sorted):
                is_first = (i == 0)
                is_last = (i == len(scenes_sorted) - 1)
                
                original_start = float(scene["start"])
                original_end = float(scene["end"])
                
                # Start: first scene = 0, others use the previous cut point
                if is_first:
                    new_start = 0.0
                    start_snap_info = {"snapped": False, "reason": "first_scene"}
                else:
                    prev_cut = cut_points[i - 1]
                    new_start = prev_cut["cut_point"]
                    start_snap_info = {
                        "snapped": True,
                        "original": original_start,
                        "snapped_to": new_start,
                        "delta_ms": round((new_start - original_start) * 1000),
                        "cut_rms_db": prev_cut["cut_rms_db"],
                        "gap_ms": prev_cut["gap_ms"],
                        "window": [prev_cut["window_start"], prev_cut["window_end"]],
                        "moved_from_midpoint_ms": prev_cut["moved_from_midpoint_ms"],
                        "speech_floor_db": prev_cut.get("speech_floor_db"),
                        "last_speech_idx": prev_cut.get("last_speech_idx"),
                    }
                
                # End: last scene = audio duration, others use next cut point
                if is_last:
                    new_end = min(original_end, audio_duration)
                    end_snap_info = {"snapped": False, "reason": "last_scene"}
                else:
                    next_cut = cut_points[i]
                    new_end = next_cut["cut_point"]
                    end_snap_info = {
                        "snapped": True,
                        "original": original_end,
                        "snapped_to": new_end,
                        "delta_ms": round((new_end - original_end) * 1000),
                        "cut_rms_db": next_cut["cut_rms_db"],
                        "gap_ms": next_cut["gap_ms"],
                        "window": [next_cut["window_start"], next_cut["window_end"]],
                        "moved_from_midpoint_ms": next_cut["moved_from_midpoint_ms"],
                        "speech_floor_db": next_cut.get("speech_floor_db"),
                        "last_speech_idx": next_cut.get("last_speech_idx"),
                    }
                
                # Sanity check
                if new_start >= new_end:
                    new_start = original_start
                    new_end = original_end
                    start_snap_info = {"snapped": False, "reason": "sanity_check_failed"}
                    end_snap_info = {"snapped": False, "reason": "sanity_check_failed"}
                
                adjusted_scenes.append({
                    "part": scene["part"],
                    "start": round(new_start, 3),
                    "end": round(new_end, 3),
                })
                
                snap_metadata.append({
                    "part": scene["part"],
                    "original_start": original_start,
                    "original_end": original_end,
                    "new_start": round(new_start, 3),
                    "new_end": round(new_end, 3),
                    "start_snap": start_snap_info,
                    "end_snap": end_snap_info,
                })
            
            # Count actual snap events
            snapped_count = 0
            fallback_count = 0
            for m in snap_metadata:
                if m["start_snap"].get("snapped"):
                    snapped_count += 1
                elif m["start_snap"].get("reason") not in ("first_scene",):
                    fallback_count += 1
                if m["end_snap"].get("snapped"):
                    snapped_count += 1
                elif m["end_snap"].get("reason") not in ("last_scene",):
                    fallback_count += 1
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            return {
                "status": "ok",
                "mode": "gap_quietest",
                "algorithm_version": "v0.11.0_post_speech",
                "audio_duration": round(audio_duration, 3),
                "buffer_ms": buffer_ms,
                "chunk_ms": chunk_ms,
                "snapped_count": snapped_count,
                "fallback_count": fallback_count,
                "total_boundaries": snapped_count + fallback_count,
                "cut_points": [
                    {
                        "between_scene": f"{scenes_sorted[i]['part']} -> {scenes_sorted[i+1]['part']}",
                        "cut_point": cp["cut_point"],
                        "cut_rms_db": cp["cut_rms_db"],
                        "gap_ms": cp["gap_ms"],
                        "window": [cp["window_start"], cp["window_end"]],
                        "speech_floor_db": cp.get("speech_floor_db"),
                        "last_speech_idx": cp.get("last_speech_idx"),
                    }
                    for i, cp in enumerate(cut_points)
                ],
                "adjusted_scenes": adjusted_scenes,
                "snap_details": snap_metadata,
                "processing_time_ms": elapsed_ms,
            }
        
        # ============================================
        # LEGACY MODES (rms_minimum, silence) — backward compat
        # ============================================
        adjusted_scenes = []
        snap_metadata = []
        snapped_count = 0
        fallback_count = 0
        
        def snap_point(target_time):
            """Snap a single timestamp using the selected legacy mode."""
            if mode == "rms_minimum":
                result = find_rms_minimum(
                    input_path, target_time, audio_duration,
                    window_s=window_s, chunk_ms=chunk_ms
                )
                return {
                    "snapped": result.get("quietest_rms_db") is not None,
                    "new_time": result["quietest_time"],
                    "info": result,
                }
            else:  # silence mode
                snap = find_nearest_silence(
                    target_time, silences, window_s, min_silence_ms
                )
                if snap:
                    return {
                        "snapped": True,
                        "new_time": round(snap["midpoint"], 3),
                        "info": {
                            "silence_used_ms": snap["silence"]["duration_ms"],
                            "distance_from_target_ms": snap["distance_ms"],
                        },
                    }
                return {
                    "snapped": False,
                    "new_time": target_time,
                    "info": {"reason": "no_silence_in_window"},
                }
        
        for i, scene in enumerate(scenes_sorted):
            is_first = (i == 0)
            is_last = (i == len(scenes_sorted) - 1)
            
            original_start = float(scene["start"])
            original_end = float(scene["end"])
            
            if is_first:
                new_start = 0.0
                start_snap_info = {"snapped": False, "reason": "first_scene"}
            else:
                snap_result = snap_point(original_start)
                new_start = round(snap_result["new_time"], 3)
                if snap_result["snapped"]:
                    start_snap_info = {
                        "snapped": True,
                        "original": original_start,
                        "snapped_to": new_start,
                        "delta_ms": round((new_start - original_start) * 1000),
                        **snap_result["info"],
                    }
                    snapped_count += 1
                else:
                    start_snap_info = {"snapped": False, **snap_result["info"]}
                    fallback_count += 1
            
            if is_last:
                new_end = min(original_end, audio_duration)
                end_snap_info = {"snapped": False, "reason": "last_scene"}
            else:
                snap_result = snap_point(original_end)
                new_end = round(snap_result["new_time"], 3)
                if snap_result["snapped"]:
                    end_snap_info = {
                        "snapped": True,
                        "original": original_end,
                        "snapped_to": new_end,
                        "delta_ms": round((new_end - original_end) * 1000),
                        **snap_result["info"],
                    }
                    snapped_count += 1
                else:
                    end_snap_info = {"snapped": False, **snap_result["info"]}
                    fallback_count += 1
            
            if new_start >= new_end:
                new_start = original_start
                new_end = original_end
                start_snap_info = {"snapped": False, "reason": "sanity_check_failed"}
                end_snap_info = {"snapped": False, "reason": "sanity_check_failed"}
            
            adjusted_scenes.append({
                "part": scene["part"],
                "start": new_start,
                "end": new_end,
            })
            snap_metadata.append({
                "part": scene["part"],
                "original_start": original_start,
                "original_end": original_end,
                "new_start": new_start,
                "new_end": new_end,
                "start_snap": start_snap_info,
                "end_snap": end_snap_info,
            })
        
        # Coordination: prevent overlaps
        for i in range(len(adjusted_scenes) - 1):
            current = adjusted_scenes[i]
            next_scene = adjusted_scenes[i + 1]
            if current["end"] > next_scene["start"]:
                midpoint = (current["end"] + next_scene["start"]) / 2
                current["end"] = round(midpoint, 3)
                next_scene["start"] = round(midpoint, 3)
                snap_metadata[i]["overlap_corrected"] = True
                snap_metadata[i + 1]["overlap_corrected"] = True
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return {
            "status": "ok",
            "mode": mode,
            "audio_duration": round(audio_duration, 3),
            "search_window_ms": search_window_ms,
            "chunk_ms": chunk_ms if mode == "rms_minimum" else None,
            "min_silence_ms": min_silence_ms if mode == "silence" else None,
            "silence_threshold_db": round(silence_threshold_db, 2) if silence_threshold_db else None,
            "total_silences_detected": len(silences) if mode == "silence" else None,
            "snapped_count": snapped_count,
            "fallback_count": fallback_count,
            "total_boundaries": snapped_count + fallback_count,
            "adjusted_scenes": adjusted_scenes,
            "snap_details": snap_metadata,
            "processing_time_ms": elapsed_ms,
        }
    
    finally:
        input_path.unlink(missing_ok=True)


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
    """
    Cut video+audio to [start, end] range with frame-accurate cuts.
    
    Uses slow seek (-ss AFTER -i) for frame-accurate decoding:
      - Decodes from start of file
      - Skips frames until start timestamp (accurate, no black frames)
      - Slower than fast seek but guarantees correct output
    
    Also forces keyframe at start to avoid black frames at beginning.
    """
    duration = end - start
    return run_ffmpeg([
        "-i", str(input_path),
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-force_key_frames", "0",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        str(output_path)
    ])


def add_audio_fade(
    input_path: Path,
    output_path: Path,
    fade_in_ms: int = 50,
    fade_out_ms: int = 50,
) -> tuple:
    """
    Apply audio-only fade-in at start and fade-out at end of video.
    
    v0.10.1: Used as final step in /video/optimize to ensure smooth audio
    transitions when scenes are concatenated downstream.
    
    Behavior:
      - Audio: fade-in over first {fade_in_ms} + fade-out over last {fade_out_ms}
      - Video: completely unchanged (stream-copied for speed)
      - Fade happens mostly over the trim buffer zone (where audio is mostly
        silent anyway), with minimal touch on speech edges.
    
    When scenes are concatenated downstream:
      Scene 1: [...content...][fade-out] | Scene 2: [fade-in][...content...]
                              ^                   ^
                              audio bridges smoothly between scenes
    
    Args:
        input_path: source video
        output_path: destination video with audio fades
        fade_in_ms: fade-in duration at start (default 50ms)
        fade_out_ms: fade-out duration at end (default 50ms)
    
    Returns:
        (returncode, stderr)
    """
    duration = get_duration(input_path)
    if duration <= 0:
        return -1, "invalid input duration"
    
    fade_in_s = fade_in_ms / 1000
    fade_out_s = fade_out_ms / 1000
    
    # Fade-out starts at (duration - fade_out_s)
    fade_out_start = max(0, duration - fade_out_s)
    
    # Audio filter: fade-in from 0, fade-out to end
    audio_filter = (
        f"afade=t=in:st=0:d={fade_in_s:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_s:.3f}"
    )
    
    # Video stream-copied (no re-encode = fast + lossless)
    # Audio re-encoded with fade filters
    return run_ffmpeg([
        "-i", str(input_path),
        "-c:v", "copy",
        "-af", audio_filter,
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
        "min_speedup": 1.07,
        "max_speedup": 1.15,
    },
}


def decide_video_optimization(
    analysis: dict,
    duration: float,
    profile: dict,
    audio_path: Optional[Path] = None
) -> dict:
    """
    Decide how to optimize the video based on audio analysis and UGC profile.
    
    If audio_path is provided, runs RMS-based trailing breath detection
    to catch puste/breath that silencedetect missed.
    
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
        # Leading: pause at the very start
        if silences[0]["start"] <= EDGE_TOLERANCE and silences[0]["duration_ms"] > EDGE_SILENCE_MIN_MS:
            leading_silence = silences[0]
        
        # Trailing: pause at/near the end
        # Strict: pause ends within 50ms of duration
        # OR: large pause (>=400ms) ends within 400ms of duration (catches
        #     cases where short speech/breath exists between pause and end)
        last_silence = silences[-1]
        if last_silence is not leading_silence:
            ends_near = last_silence["end"] >= duration - EDGE_TOLERANCE
            is_long_near_end = (
                last_silence["duration_ms"] >= 400
                and last_silence["end"] >= duration - 0.4
            )
            has_min_duration = last_silence["duration_ms"] > EDGE_SILENCE_MIN_MS
            
            if has_min_duration and (ends_near or is_long_near_end):
                trailing_silence = last_silence
    
    # Calculate cut boundaries (asymmetric: 50ms leading / 25ms trailing)
    leading_keep_s = EDGE_SILENCE_KEEP_LEADING_MS / 1000
    trailing_keep_s = EDGE_SILENCE_KEEP_TRAILING_MS / 1000
    
    if leading_silence:
        trim_start = max(0, leading_silence["end"] - leading_keep_s)
    else:
        trim_start = 0.0
    
    if trailing_silence:
        trim_end = min(duration, trailing_silence["start"] + trailing_keep_s)
    else:
        trim_end = duration
    
    # v0.9.1: Smart trailing breath detection via RMS scan
    # Only trigger if silencedetect found NO trailing silence (likely missed breath)
    breath_trim_applied = False
    breath_trimmed_ms = 0
    original_trim_end = trim_end
    
    if audio_path is not None and trailing_silence is None:
        real_speech_end = detect_trailing_breath(audio_path, analysis, scan_duration=1.0)
        if real_speech_end < duration - 0.05:
            # Found breath tail - trim it (use tighter trailing buffer)
            proposed_trim_end = min(duration, real_speech_end + trailing_keep_s)
            if proposed_trim_end < trim_end:
                breath_trimmed_ms = round((trim_end - proposed_trim_end) * 1000)
                trim_end = proposed_trim_end
                breath_trim_applied = True
    
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
    # too_slow if EITHER density is below target OR duration is above target
    if trimmed_density < target_density * 0.95 or duration_ratio > 1.05:
        classification = "too_slow"
    elif duration_ratio < 0.85 and trimmed_density > target_density * 1.05:
        classification = "too_fast"
    else:
        classification = "good"
    
    # ============================================
    # EMPHASIS DETECTION (v0.9.3)
    # Distinguishes deliberate performance from actual slow delivery
    # ============================================
    emphasis_score = 0
    emphasis_reasons = []
    
    if mid_silences:
        mid_pause_durations = [s["duration_ms"] for s in mid_silences]
        max_mid_pause = max(mid_pause_durations)
        avg_pause_calc = sum(mid_pause_durations) / len(mid_pause_durations)
        
        # Signal 1: Pause count (fewer = more deliberate)
        if len(mid_silences) <= 2:
            emphasis_score += 2
            emphasis_reasons.append(f"few pauses ({len(mid_silences)})")
        elif len(mid_silences) <= 3:
            emphasis_score += 2
            emphasis_reasons.append(f"moderate pause count ({len(mid_silences)})")
        elif len(mid_silences) <= 4:
            emphasis_score += 1
            emphasis_reasons.append(f"balanced pauses ({len(mid_silences)})")
        
        # Signal 2: Average pause duration (long avg = deliberate pacing)
        if avg_pause_calc >= 400:
            emphasis_score += 3
            emphasis_reasons.append(f"very long avg pause ({avg_pause_calc:.0f}ms)")
        elif avg_pause_calc >= 300:
            emphasis_score += 2
            emphasis_reasons.append(f"long avg pause ({avg_pause_calc:.0f}ms)")
        elif avg_pause_calc >= 200:
            emphasis_score += 1
            emphasis_reasons.append(f"moderate avg pause ({avg_pause_calc:.0f}ms)")
        
        # Signal 3: Dominant pause (single emphasis beat stands out)
        if avg_pause_calc > 0 and max_mid_pause / avg_pause_calc >= 1.5 and max_mid_pause >= 300:
            emphasis_score += 2
            emphasis_reasons.append(f"dominant pause ({max_mid_pause}ms vs avg {avg_pause_calc:.0f}ms)")
        
        # Signal 4: Weighted dramatic pauses (by duration, not just count)
        dramatic_score = 0
        for p in mid_pause_durations:
            if p >= 600:
                dramatic_score += 3    # very dramatic
            elif p >= 450:
                dramatic_score += 2    # clearly dramatic
            elif p >= 350:
                dramatic_score += 1    # mildly dramatic
        
        if dramatic_score > 0:
            emphasis_score += dramatic_score
            count_dramatic = len([p for p in mid_pause_durations if p >= 350])
            emphasis_reasons.append(f"{count_dramatic} dramatic pause(s) weighted +{dramatic_score}")
    
    # Signal 5: Penalty for flat/stretched delivery
    # True emphatic delivery has tight speech between pauses
    # Stretched delivery has low density AND low onset rate even with long pauses
    style = analysis["speech"]["style"]
    
    # Strong penalty: slow style + low onset + low density = definitely stretched
    # (e.g. Seedance artificial stretching with random long pause)
    if style == "slow" and onset_rate < 1.0 and trimmed_density < 0.80:
        emphasis_score -= 6
        emphasis_reasons.append(
            f"STRETCHED penalty -6: slow + onset {onset_rate:.2f} + density {trimmed_density:.2f}"
        )
    # Moderate penalty: flat slow delivery without other strong signals
    elif style == "slow" and onset_rate < 1.0 and emphasis_score < 3:
        emphasis_score -= 2
        emphasis_reasons.append(f"flat slow delivery (onset rate {onset_rate:.2f})")
    
    # Threshold: Need score >= 4 for emphatic classification
    is_emphatic = emphasis_score >= 4
    
    # Smart override based on emphasis confidence + duration
    # Higher emphasis score = allow longer duration before overriding
    # Score 4-6:  override if > 1.20x target (moderate confidence)
    # Score 7+:   override if > 1.35x target (high confidence in emphasis)
    if is_emphatic:
        if emphasis_score >= 7:
            override_threshold = target_duration * 1.35  # generous for strong emphasis
        else:
            override_threshold = target_duration * 1.20  # moderate for weak emphasis
        
        if trimmed_duration > override_threshold:
            is_emphatic = False
            emphasis_reasons.append(
                f"OVERRIDE: duration {trimmed_duration:.2f}s > {override_threshold:.2f}s "
                f"(threshold scaled by emphasis score {emphasis_score}), "
                f"treating as Seedance stretching"
            )
    
    # Calculate speedup (ALWAYS apply at least min_speedup unless too_fast)
    min_speedup_val = profile.get("min_speedup", 1.0)
    max_speedup_val = profile["max_speedup"]
    
    if classification == "too_fast":
        # Never speed up an already-fast file
        speedup = 1.0
        speedup_reason = "already fast enough, no speedup applied"
    elif classification == "too_slow":
        if is_emphatic:
            # DELIBERATE performance with emphasis beats - respect the delivery
            # Fixed gentle speedup - NO gradient for emphatic clips
            # (preserves emphasis beats, natural delivery)
            speedup = min_speedup_val
            speedup_reason = (
                f"emphatic delivery (score {emphasis_score}): "
                f"{', '.join(emphasis_reasons)}. Gentle {min_speedup_val}x boost."
            )
        else:
            # Gradient speedup based on how far over target
            # Smooth interpolation between min and max speedup
            if trimmed_duration <= target_duration * 1.05:
                # Within 5% of target - just baseline boost
                speedup = min_speedup_val
                speedup_reason = (
                    f"too_slow but close to target ({trimmed_duration:.2f}s): "
                    f"gentle {min_speedup_val}x boost"
                )
            else:
                # Calculate ideal speedup (what would bring exactly to target)
                ideal_speedup = trimmed_duration / target_duration
                
                # Clamp to min/max range
                speedup = max(min_speedup_val, min(ideal_speedup, max_speedup_val))
                
                speedup_reason = (
                    f"too_slow gradient: {trimmed_duration:.2f}s needs {ideal_speedup:.3f}x "
                    f"to hit target, clamped to {speedup:.3f}x (range {min_speedup_val}-{max_speedup_val})"
                )
        
        speedup = round(speedup, 3)
    else:
        # "good" classification - apply min_speedup for consistent UGC pace
        speedup = round(min_speedup_val, 3)
        speedup_reason = f"baseline boost {min_speedup_val}x for UGC pace"
    
    final_duration = trimmed_duration / speedup if speedup > 1.0 else trimmed_duration
    
    return {
        "trim_start": round(trim_start, 3),
        "trim_end": round(trim_end, 3),
        "trimmed_duration": round(trimmed_duration, 3),
        "leading_silence_ms": leading_silence["duration_ms"] if leading_silence else 0,
        "trailing_silence_ms": trailing_silence["duration_ms"] if trailing_silence else 0,
        "breath_trim_applied": breath_trim_applied,
        "breath_trimmed_ms": breath_trimmed_ms,
        "mid_silence_ms": round(mid_silence_ms),
        "avg_mid_pause_ms": round(avg_mid_pause_ms),
        "trimmed_density": round(trimmed_density, 3),
        "onset_rate": round(onset_rate, 2),
        "emphasis_score": emphasis_score,
        "emphasis_reasons": emphasis_reasons,
        "is_emphatic": is_emphatic,
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


# ============================================
# NEW v0.9.6: VIDEO MERGE-AUDIO ENDPOINT
# ============================================
@app.post("/video/merge-audio")
async def video_merge_audio_endpoint(
    video: UploadFile = File(..., description="Silent video file (BROLL from Seedance)"),
    audio_url: Optional[str] = Form(None, description="URL to audio file (VO from MinIO)"),
    audio_file: Optional[UploadFile] = File(None, description="Alternative: direct audio upload"),
    auth: str = Depends(verify_token)
):
    """
    Merge a silent video with an audio track.
    
    Use case: BROLL scenes generated WITHOUT audio by Seedance 2.0
    (generate_audio: false). This endpoint attaches the ElevenLabs VO
    as audio track before the video goes to /video/optimize.
    
    Audio source priority: audio_file (upload) > audio_url (download).
    
    Behavior:
      - Audio shorter than video: audio plays, video continues silent to end
      - Audio longer than video:  audio is cut to video duration
      - Video stream: copied (no re-encode - fast + lossless)
      - Audio stream: encoded as AAC 192kbps
    
    Form params:
      video:      Silent video file (MP4) — required
      audio_url:  Optional URL to download audio from (e.g. MinIO)
      audio_file: Optional direct audio file upload (alternative to audio_url)
    
    Returns merged video as base64 + metadata.
    
    Pipeline position:
      Seedance (silent BROLL) → /video/merge-audio → /video/optimize → final
    """
    if not audio_url and not audio_file:
        raise HTTPException(400, "Must provide either audio_url or audio_file")
    
    start_time = time.time()
    job_id = str(uuid.uuid4())[:8]
    video_path = WORK_DIR / f"{job_id}_video_in.mp4"
    audio_path = WORK_DIR / f"{job_id}_audio_in.mp3"
    output_path = WORK_DIR / f"{job_id}_merged.mp4"
    
    try:
        # Save video
        video_content = await video.read()
        with open(video_path, "wb") as f:
            f.write(video_content)
        
        # Get audio: prefer uploaded file, fall back to URL download
        if audio_file:
            audio_content = await audio_file.read()
            with open(audio_path, "wb") as f:
                f.write(audio_content)
            audio_source = "upload"
        else:
            # Download audio from URL
            try:
                urllib.request.urlretrieve(audio_url, audio_path)
                audio_source = "url_download"
            except Exception as e:
                raise HTTPException(400, f"Failed to download audio from URL: {e}")
        
        # Get durations
        video_duration = get_duration(video_path)
        audio_duration = get_duration(audio_path)
        
        if video_duration <= 0:
            raise HTTPException(400, "Invalid video file (zero duration)")
        if audio_duration <= 0:
            raise HTTPException(400, "Invalid audio file (zero duration)")
        
        # Build merge command
        # -map 0:v:0  → video from input 0
        # -map 1:a:0  → audio from input 1
        # -c:v copy   → don't re-encode video (fast + lossless)
        # -c:a aac    → encode audio as AAC
        # If audio longer than video, cap at video duration with -t
        cmd = [
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
        ]
        
        audio_was_cut = audio_duration > video_duration
        if audio_was_cut:
            cmd.extend(["-t", f"{video_duration:.3f}"])
        
        cmd.append(str(output_path))
        
        ret, err = run_ffmpeg(cmd)
        if ret != 0:
            return {"status": "error", "error": f"merge failed: {err[:200]}"}
        
        output_duration = get_duration(output_path)
        
        # Read merged video
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return {
            "status": "ok",
            "audio_source": audio_source,
            "video_duration": round(video_duration, 3),
            "audio_duration": round(audio_duration, 3),
            "output_duration": round(output_duration, 3),
            "audio_was_cut": audio_was_cut,
            "audio_shorter_than_video": audio_duration < video_duration,
            "video_base64": video_b64,
            "size_bytes": len(video_bytes),
            "processing_time_ms": elapsed_ms
        }
    
    finally:
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


# ============================================
# /video/optimize (UNCHANGED from v0.9.5)
# ============================================
@app.post("/video/optimize")
async def video_optimize_endpoint(
    file: UploadFile = File(...),
    target_duration: Optional[float] = Form(None),
    min_speedup: Optional[float] = Form(None),
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
      3. Apply: trim edges, then speedup (always at least min_speedup)
      4. Return optimized video + decision details
    
    Default profile:
      - target_final_duration: 3.0s
      - target_density: 0.85
      - min_speedup: 1.07x (always applied for UGC pace)
      - max_speedup: 1.15x (cap for too_slow files)
      - max_pause_ms: 250
    
    Speedup logic:
      - too_slow:  between min_speedup and max_speedup (based on duration)
      - good:      min_speedup baseline (for consistent UGC pace)
      - too_fast:  no speedup (1.0x)
    
    Form params:
      file:              Video file (MP4)
      target_duration:   Optional override for target duration
      min_speedup:       Optional override for baseline speedup
      max_speedup:       Optional override for max speedup cap
      apply_trim:        If True, trim leading/trailing silence (default: True)
      apply_speedup:     If True, apply speedup (default: True)
    """
    profile_settings = dict(UGC_PROFILES["default"])
    if target_duration is not None:
        profile_settings["target_final_duration"] = target_duration
    if min_speedup is not None:
        profile_settings["min_speedup"] = min_speedup
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
        
        # Step 2: Decide optimizations based on profile (uses audio_path for breath detection)
        decision = decide_video_optimization(analysis, video_duration, profile_settings, audio_path=audio_path)
        
        # Audio path no longer needed after decision
        audio_path.unlink(missing_ok=True)
        
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
        
        # If no operation applied, use original (need to copy for next step)
        if current_path == video_path:
            shutil.copy(video_path, final_path)
            current_path = final_path
        elif current_path != final_path:
            # Move cut result to final
            current_path.rename(final_path)
            current_path = final_path
        
        # Step 5: Apply audio fade-in/fade-out (v0.10.1)
        # Audio-only: video remains unchanged (stream-copied)
        # Ensures smooth audio at scene boundaries when concatenated downstream
        faded_path = WORK_DIR / f"{job_id}_faded.mp4"
        ret, err = add_audio_fade(
            final_path, faded_path,
            fade_in_ms=50,
            fade_out_ms=50,
        )
        if ret != 0:
            return {"status": "error", "error": f"audio fade failed: {err[:200]}"}
        
        final_path.unlink(missing_ok=True)
        faded_path.rename(final_path)
        
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


# ============================================
# v0.9.12 — VIDEO PRESET (color grading)
# ============================================
# Lightroom-style color grading filter for video clips.
# Apply consistent visual treatment across all UGC scenes.
# ============================================

# Default UGC preset values (Lightroom-style scale: -100 to +100)
DEFAULT_PRESET = {
    "temp":       -3,
    "tint":       2,
    "saturation": -6,
    "exposure":   -3,
    "contrast":   12,
    "highlight":  -35,
    "shadow":     18,
    "fade":       6,
}


def build_color_filter_chain(preset: dict) -> str:
    """
    Build ffmpeg filter chain from Lightroom-style preset values.
    
    Mappings (Lightroom -100/+100 → ffmpeg specific):
      - exposure: brightness offset (-1 to +1, scaled by 0.01)
      - contrast: contrast multiplier (1.0 = no change, ±0.01 per unit)
      - saturation: saturation multiplier (1.0 = no change, ±0.01 per unit)
      - temp: cool/warm via b-channel gamma + colorbalance
              Negative = cooler (more blue), positive = warmer
      - tint: magenta/green via colorbalance
              Negative = green, positive = magenta
      - highlight: reduce highlights via curves
      - shadow: lift shadows via curves
      - fade: matte look (raise blacks + lower whites)
    """
    filters = []
    
    # === STEP 1: Exposure + Contrast + Saturation (eq filter) ===
    # exposure: -100..+100 → brightness ±0.5
    brightness = preset.get("exposure", 0) * 0.005
    # contrast: -100..+100 → multiplier 0.5..1.5
    contrast = 1.0 + (preset.get("contrast", 0) * 0.01)
    # saturation: -100..+100 → multiplier 0.0..2.0
    saturation = 1.0 + (preset.get("saturation", 0) * 0.01)
    
    eq_parts = []
    if abs(brightness) > 0.001:
        eq_parts.append(f"brightness={brightness:.3f}")
    if abs(contrast - 1.0) > 0.001:
        eq_parts.append(f"contrast={contrast:.3f}")
    if abs(saturation - 1.0) > 0.001:
        eq_parts.append(f"saturation={saturation:.3f}")
    
    if eq_parts:
        filters.append(f"eq={':'.join(eq_parts)}")
    
    # === STEP 2: Temp + Tint (colorbalance filter) ===
    # temp: -100..+100 → blue-yellow shift (-/+)
    #   Negative temp = add blue → bs (blue shadow) +
    #   Positive temp = add yellow → rs (red shadow) + gs minor
    # tint: -100..+100 → green-magenta shift
    #   Negative tint = green → gs +
    #   Positive tint = magenta → rs + bs
    temp = preset.get("temp", 0)
    tint = preset.get("tint", 0)
    
    if abs(temp) > 0 or abs(tint) > 0:
        # Scale: ±100 lightroom → ±0.5 colorbalance value (subtle effect)
        # Apply mostly on midtones (gm) for natural look
        rm = (temp * 0.003) + (tint * 0.003)         # red midtones
        gm = -(tint * 0.005)                          # green midtones (inverse for magenta)
        bm = -(temp * 0.005) + (tint * 0.002)        # blue midtones (inverse for warm)
        
        cb_parts = []
        if abs(rm) > 0.001:
            cb_parts.append(f"rm={rm:.3f}")
        if abs(gm) > 0.001:
            cb_parts.append(f"gm={gm:.3f}")
        if abs(bm) > 0.001:
            cb_parts.append(f"bm={bm:.3f}")
        
        if cb_parts:
            filters.append(f"colorbalance={':'.join(cb_parts)}")
    
    # === STEP 3: Highlight + Shadow + Fade (curves filter) ===
    # Build a curve from anchor points
    highlight = preset.get("highlight", 0)
    shadow = preset.get("shadow", 0)
    fade = preset.get("fade", 0)
    
    if abs(highlight) > 0 or abs(shadow) > 0 or abs(fade) > 0:
        # Base curve points (input → output)
        # Default linear: 0,0 → 0.25,0.25 → 0.5,0.5 → 0.75,0.75 → 1,1
        
        # Black point (0,0): lift by fade
        # Lightroom fade +100 → black point ~0.15
        black_in = 0.0
        black_out = max(0.0, fade * 0.0015)  # +6 fade → +0.009 (subtle lift)
        
        # Shadow point (0.25, 0.25): lift by shadow
        # +18 shadow → +0.045 lift (subtle)
        shadow_in = 0.25
        shadow_out = 0.25 + (shadow * 0.0025)
        shadow_out = max(0.0, min(1.0, shadow_out))
        
        # Highlight point (0.75, 0.75): pull down by highlight
        # -35 highlight → -0.0875 (clear reduction)
        highlight_in = 0.75
        highlight_out = 0.75 + (highlight * 0.0025)  # negative = pull down
        highlight_out = max(0.0, min(1.0, highlight_out))
        
        # White point (1,1): crushed by fade
        white_in = 1.0
        white_out = min(1.0, 1.0 - (fade * 0.0010))  # +6 fade → -0.006 crush
        
        # Build curves all_channels for master tonal curve
        curve_points = (
            f"{black_in:.3f}/{black_out:.3f} "
            f"{shadow_in:.3f}/{shadow_out:.3f} "
            f"{highlight_in:.3f}/{highlight_out:.3f} "
            f"{white_in:.3f}/{white_out:.3f}"
        )
        filters.append(f"curves=all='{curve_points}'")
    
    # === STEP 4: Output format compatibility ===
    filters.append("format=yuv420p")
    
    return ",".join(filters) if filters else "null"


@app.post("/video/preset")
async def video_preset_endpoint(
    file: UploadFile = File(...),
    temp: float = Form(DEFAULT_PRESET["temp"]),
    tint: float = Form(DEFAULT_PRESET["tint"]),
    saturation: float = Form(DEFAULT_PRESET["saturation"]),
    exposure: float = Form(DEFAULT_PRESET["exposure"]),
    contrast: float = Form(DEFAULT_PRESET["contrast"]),
    highlight: float = Form(DEFAULT_PRESET["highlight"]),
    shadow: float = Form(DEFAULT_PRESET["shadow"]),
    fade: float = Form(DEFAULT_PRESET["fade"]),
    auth: str = Depends(verify_token)
):
    """
    Apply Lightroom-style color grading to video.
    
    Settings use Lightroom scale (-100 to +100) for intuitive control.
    Defaults match the standard UGC look (modern faded TikTok aesthetic):
      temp=-3, tint=+2, saturation=-6, exposure=-3,
      contrast=+12, highlight=-35, shadow=+18, fade=+6
    
    All parameters are optional — sending no params applies the default UGC preset.
    Audio stream is copied unchanged (no re-encode).
    
    Form params (all optional, all -100 to +100):
      file:       Video file (MP4)
      temp:       Color temperature (- = cool/blue, + = warm/yellow)
      tint:       Color tint (- = green, + = magenta)
      saturation: Color saturation (- = muted, + = punchy)
      exposure:   Brightness (- = darker, + = brighter)
      contrast:   Contrast (- = flat, + = punchy)
      highlight:  Highlight recovery (- = pull down highlights)
      shadow:     Shadow lift (+ = lift dark areas)
      fade:       Matte/faded look (+ = lift blacks, crush whites)
    
    Pipeline position:
      Final step before upload — applied AFTER /video/optimize
      Flow: ... → /video/optimize → /video/preset → upload
    
    Returns: Color-graded video as base64 + applied preset values
    """
    preset = {
        "temp":       temp,
        "tint":       tint,
        "saturation": saturation,
        "exposure":   exposure,
        "contrast":   contrast,
        "highlight":  highlight,
        "shadow":     shadow,
        "fade":       fade,
    }
    
    start_time = time.time()
    job_id = str(uuid.uuid4())[:8]
    video_path = WORK_DIR / f"{job_id}_input.mp4"
    output_path = WORK_DIR / f"{job_id}_graded.mp4"
    
    try:
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        original_duration = get_duration(video_path)
        
        if original_duration <= 0:
            raise HTTPException(400, "Invalid video file (zero duration)")
        
        # Build the filter chain from preset values
        filter_chain = build_color_filter_chain(preset)
        
        # Apply color grading
        # Video: re-encode with filter chain
        # Audio: copy unchanged (much faster)
        ret, err = run_ffmpeg([
            "-i", str(video_path),
            "-vf", filter_chain,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path)
        ])
        
        if ret != 0:
            return {"status": "error", "error": f"color grading failed: {err[:300]}"}
        
        final_duration = get_duration(output_path)
        
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Determine if any non-default values are used
        is_default = all(preset[k] == DEFAULT_PRESET[k] for k in DEFAULT_PRESET)
        
        return {
            "status": "ok",
            "preset_used": preset,
            "is_default_preset": is_default,
            "filter_chain_applied": filter_chain,
            "original_duration": round(original_duration, 3),
            "final_duration": round(final_duration, 3),
            "video_base64": video_b64,
            "size_bytes": len(video_bytes),
            "processing_time_ms": elapsed_ms
        }
    
    finally:
        video_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
