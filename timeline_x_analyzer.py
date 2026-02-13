"""
TIMELINE X - MEDIA ANALYZER
============================

This module handles the actual analysis of video and audio files:
- FFprobe for video metadata (duration, resolution, frame rate, codec)
- Librosa for BPM detection and beat mapping
- Audio waveform analysis for energy curves
- Integration with the framework's ClipAnalysis and AudioAnalysis dataclasses

By Aerial Canvas
"""

import os
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import math

# Import framework classes
try:
    from timeline_x_framework import (
        ShotType, CameraMovement, ContentCategory, TechnicalQuality,
        ClipAnalysis, AudioAnalysis, BeatMap
    )
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False

# Check for librosa availability
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ============================================================================
# FFPROBE ANALYZER - Video Metadata Extraction
# ============================================================================

class FFProbeAnalyzer:
    """
    Extracts video metadata using FFprobe.
    FFprobe comes bundled with FFmpeg.
    """

    def __init__(self):
        self.ffprobe_path = self._find_ffprobe()

    def _find_ffprobe(self) -> Optional[str]:
        """Find ffprobe in system PATH or common locations"""
        # Try system PATH first
        try:
            result = subprocess.run(
                ["which", "ffprobe"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

        # Common installation paths
        common_paths = [
            "/usr/local/bin/ffprobe",
            "/usr/bin/ffprobe",
            "/opt/homebrew/bin/ffprobe",  # Apple Silicon Homebrew
            "/opt/local/bin/ffprobe",      # MacPorts
        ]

        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path

        return None

    def is_available(self) -> bool:
        """Check if FFprobe is available"""
        return self.ffprobe_path is not None

    def analyze_video(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a video file and extract metadata.

        Returns dict with:
        - duration: float (seconds)
        - width: int
        - height: int
        - frame_rate: float
        - codec: str
        - has_audio: bool
        - audio_codec: str (if audio present)
        - audio_sample_rate: int (if audio present)
        """
        if not self.ffprobe_path or not os.path.exists(file_path):
            return None

        try:
            cmd = [
                self.ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)

            # Extract video stream info
            video_stream = None
            audio_stream = None

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video" and video_stream is None:
                    video_stream = stream
                elif stream.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = stream

            if not video_stream:
                return None

            # Parse frame rate (can be "24000/1001" format)
            frame_rate_str = video_stream.get("r_frame_rate", "24/1")
            if "/" in frame_rate_str:
                num, den = frame_rate_str.split("/")
                frame_rate = float(num) / float(den) if float(den) != 0 else 24.0
            else:
                frame_rate = float(frame_rate_str)

            # Get duration from format or stream
            duration = float(data.get("format", {}).get("duration", 0))
            if duration == 0:
                duration = float(video_stream.get("duration", 0))

            result_data = {
                "duration": duration,
                "width": int(video_stream.get("width", 1920)),
                "height": int(video_stream.get("height", 1080)),
                "frame_rate": round(frame_rate, 3),
                "codec": video_stream.get("codec_name", "unknown"),
                "has_audio": audio_stream is not None,
                "bit_rate": int(data.get("format", {}).get("bit_rate", 0)),
                "file_size": int(data.get("format", {}).get("size", 0)),
            }

            if audio_stream:
                result_data["audio_codec"] = audio_stream.get("codec_name", "unknown")
                result_data["audio_sample_rate"] = int(audio_stream.get("sample_rate", 48000))
                result_data["audio_channels"] = int(audio_stream.get("channels", 2))

            return result_data

        except Exception as e:
            print(f"FFprobe error for {file_path}: {e}")
            return None

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Extract audio from video file for analysis.
        Returns path to extracted audio file.
        """
        if not self.ffprobe_path:
            return None

        # Find ffmpeg (should be alongside ffprobe)
        ffmpeg_path = self.ffprobe_path.replace("ffprobe", "ffmpeg")
        if not os.path.exists(ffmpeg_path):
            return None

        if output_path is None:
            # Create temp file
            output_path = tempfile.mktemp(suffix=".wav")

        try:
            cmd = [
                ffmpeg_path,
                "-y",  # Overwrite
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",
                "-ar", "22050",  # 22kHz for librosa
                "-ac", "1",  # Mono
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=60
            )

            if result.returncode == 0 and os.path.exists(output_path):
                return output_path

            return None

        except Exception as e:
            print(f"Audio extraction error: {e}")
            return None


# ============================================================================
# BPM ANALYZER - Beat Detection and Mapping
# ============================================================================

class BPMAnalyzer:
    """
    Analyzes audio for BPM, beat positions, and energy curves.
    Uses librosa if available, falls back to basic analysis otherwise.
    """

    def __init__(self):
        self.librosa_available = LIBROSA_AVAILABLE

    def is_available(self) -> bool:
        """Check if BPM analysis is available"""
        return self.librosa_available

    def analyze_audio(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze audio file for BPM and beat positions.

        Returns dict with:
        - bpm: float
        - beat_positions: List[float] (timestamps in seconds)
        - downbeats: List[float] (beat 1 of each measure)
        - energy_curve: List[Tuple[float, float]] (timestamp, energy 0-1)
        """
        if not self.librosa_available:
            return self._fallback_analysis(audio_path)

        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)

            # Detect tempo and beats
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

            # Handle tempo as array (newer librosa versions)
            if hasattr(tempo, '__len__'):
                bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                bpm = float(tempo)

            # Estimate downbeats (every 4 beats for 4/4 time)
            downbeats = beat_times[::4] if len(beat_times) >= 4 else beat_times

            # Calculate energy curve (RMS energy over time)
            hop_length = 512
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

            # Normalize energy to 0-1
            if len(rms) > 0:
                rms_min, rms_max = rms.min(), rms.max()
                if rms_max > rms_min:
                    rms_normalized = (rms - rms_min) / (rms_max - rms_min)
                else:
                    rms_normalized = np.ones_like(rms) * 0.5
            else:
                rms_normalized = np.array([0.5])
                times = np.array([0.0])

            # Sample energy curve (don't need every frame)
            sample_interval = max(1, len(times) // 100)  # ~100 points
            energy_curve = [
                (float(times[i]), float(rms_normalized[i]))
                for i in range(0, len(times), sample_interval)
            ]

            # Detect phrase boundaries (every 8 beats typically)
            phrase_boundaries = beat_times[::8] if len(beat_times) >= 8 else beat_times[::4]

            return {
                "bpm": round(bpm, 1),
                "duration": duration,
                "beat_positions": beat_times,
                "downbeats": downbeats,
                "phrase_boundaries": phrase_boundaries,
                "energy_curve": energy_curve,
                "time_signature": "4/4",  # Default assumption
                "total_beats": len(beat_times),
            }

        except Exception as e:
            print(f"Librosa analysis error: {e}")
            return self._fallback_analysis(audio_path)

    def _fallback_analysis(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Basic fallback analysis when librosa isn't available.
        Uses FFprobe for duration and estimates BPM.
        """
        ffprobe = FFProbeAnalyzer()
        if not ffprobe.is_available():
            return None

        # Get basic info via ffprobe
        try:
            cmd = [
                ffprobe.ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)
            duration = float(data.get("format", {}).get("duration", 0))

            # Return placeholder data - user should input BPM manually
            return {
                "bpm": 0.0,  # Unknown - needs manual input
                "duration": duration,
                "beat_positions": [],
                "downbeats": [],
                "phrase_boundaries": [],
                "energy_curve": [(0.0, 0.5), (duration, 0.5)],
                "time_signature": "4/4",
                "total_beats": 0,
                "needs_manual_bpm": True,
            }

        except Exception:
            return None

    def generate_beat_grid(self, bpm: float, duration: float, start_offset: float = 0.0) -> Dict[str, Any]:
        """
        Generate a mathematical beat grid from BPM.
        Useful when auto-detection fails or user provides BPM manually.
        """
        if bpm <= 0 or duration <= 0:
            return None

        beat_interval = 60.0 / bpm  # Seconds per beat

        # Generate all beats
        beat_positions = []
        current_time = start_offset
        while current_time < duration:
            beat_positions.append(round(current_time, 4))
            current_time += beat_interval

        # Downbeats (every 4 beats)
        downbeats = beat_positions[::4]

        # Phrase boundaries (every 8 beats / 2 bars)
        phrase_boundaries = beat_positions[::8]

        # Eighth notes
        eighth_interval = beat_interval / 2
        eighth_notes = []
        current_time = start_offset
        while current_time < duration:
            eighth_notes.append(round(current_time, 4))
            current_time += eighth_interval

        return {
            "bpm": bpm,
            "duration": duration,
            "beat_positions": beat_positions,
            "downbeats": downbeats,
            "phrase_boundaries": phrase_boundaries,
            "eighth_notes": eighth_notes,
            "time_signature": "4/4",
            "total_beats": len(beat_positions),
        }


# ============================================================================
# CLIP ANALYZER - Full Video Analysis Pipeline
# ============================================================================

class ClipAnalyzer:
    """
    Full video clip analysis pipeline.
    Combines FFprobe metadata with optional AI-based content analysis.
    """

    def __init__(self):
        self.ffprobe = FFProbeAnalyzer()
        self.bpm_analyzer = BPMAnalyzer()

    def analyze_clip(self, file_path: str, clip_id: Optional[str] = None) -> Optional[ClipAnalysis]:
        """
        Analyze a video clip and return a ClipAnalysis object.
        """
        if not FRAMEWORK_AVAILABLE:
            print("Framework not available")
            return None

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        # Get metadata via FFprobe
        metadata = self.ffprobe.analyze_video(file_path)
        if not metadata:
            print(f"Could not analyze: {file_path}")
            return None

        filename = os.path.basename(file_path)

        # Create ClipAnalysis object
        analysis = ClipAnalysis(
            clip_id=clip_id or filename.replace(".", "_"),
            filename=filename,
            file_path=file_path,
            total_duration=metadata["duration"],
            usable_start=0.0,
            usable_end=metadata["duration"],
            usable_duration=metadata["duration"],
            # Default classifications - can be refined by AI or user
            shot_type=ShotType.MEDIUM,
            camera_movement=CameraMovement.STATIC,
            content_category=ContentCategory.BROLL,
            technical_score=80,
            usability_score=80,
            visual_interest=70,
            has_audio=metadata.get("has_audio", False),
        )

        return analysis

    def analyze_folder(self, folder_path: str) -> List[ClipAnalysis]:
        """
        Analyze all video clips in a folder.
        """
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.prores', '.r3d', '.braw', '.m4v'}
        clips = []

        for root, dirs, files in os.walk(folder_path):
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext in video_extensions:
                    file_path = os.path.join(root, filename)
                    analysis = self.analyze_clip(file_path)
                    if analysis:
                        clips.append(analysis)

        return clips

    def analyze_music(self, file_path: str) -> Optional[AudioAnalysis]:
        """
        Analyze a music track for BPM, beats, and structure.
        """
        if not FRAMEWORK_AVAILABLE:
            return None

        if not os.path.exists(file_path):
            return None

        # Analyze with BPM analyzer
        analysis_data = self.bpm_analyzer.analyze_audio(file_path)
        if not analysis_data:
            return None

        filename = os.path.basename(file_path)

        # Create AudioAnalysis object
        audio_analysis = AudioAnalysis(
            audio_id=filename.replace(".", "_"),
            filename=filename,
            file_path=file_path,
            duration=analysis_data["duration"],
            audio_type="music",
            bpm=analysis_data["bpm"],
            time_signature=analysis_data["time_signature"],
            beat_positions=analysis_data["beat_positions"],
            downbeat_positions=analysis_data.get("downbeats", []),
            phrase_boundaries=analysis_data.get("phrase_boundaries", []),
            energy_curve=analysis_data.get("energy_curve", []),
        )

        return audio_analysis

    def create_beat_map(self, audio_analysis: AudioAnalysis) -> BeatMap:
        """
        Create a BeatMap from AudioAnalysis for timeline synchronization.
        """
        return BeatMap(
            bpm=audio_analysis.bpm,
            time_signature=audio_analysis.time_signature,
            all_beats=audio_analysis.beat_positions,
            downbeats=audio_analysis.downbeat_positions,
            phrase_boundaries=audio_analysis.phrase_boundaries,
            energy_curve=audio_analysis.energy_curve,
        )


# ============================================================================
# CONTENT ANALYZER - AI-Powered Scene Understanding (Placeholder)
# ============================================================================

class ContentAnalyzer:
    """
    Placeholder for AI-powered content analysis.
    Future integration with Director X AI for:
    - Shot type detection
    - Room/scene classification
    - Subject detection
    - Emotional tone analysis
    """

    def __init__(self):
        self.model_available = False

    def analyze_frame(self, frame) -> Dict[str, Any]:
        """Analyze a single frame for content"""
        # Placeholder - would use Director X or similar AI
        return {
            "shot_type": ShotType.MEDIUM,
            "detected_subjects": [],
            "detected_objects": [],
            "scene_type": "unknown",
            "emotional_tone": "neutral",
        }

    def classify_room(self, frame) -> str:
        """Classify room type for real estate content"""
        # Placeholder - would use CLIP or similar
        return "unknown"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_dependencies() -> Dict[str, bool]:
    """Check which analysis dependencies are available"""
    ffprobe = FFProbeAnalyzer()
    bpm = BPMAnalyzer()

    return {
        "ffprobe": ffprobe.is_available(),
        "ffprobe_path": ffprobe.ffprobe_path,
        "librosa": bpm.librosa_available,
        "framework": FRAMEWORK_AVAILABLE,
    }


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{minutes:02d}:{secs:06.3f}"


def format_timecode(seconds: float, frame_rate: float = 24.0) -> str:
    """Format seconds as timecode HH:MM:SS:FF"""
    frames = int(seconds * frame_rate)
    total_seconds = frames // int(frame_rate)
    remaining_frames = frames % int(frame_rate)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{remaining_frames:02d}"


# ============================================================================
# MAIN - Testing
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Timeline X - Media Analyzer")
    print("  Dependency Check")
    print("=" * 60 + "\n")

    deps = check_dependencies()

    print("Dependencies:")
    print(f"  FFprobe:   {'Available' if deps['ffprobe'] else 'NOT FOUND'}")
    if deps['ffprobe']:
        print(f"             Path: {deps['ffprobe_path']}")
    print(f"  Librosa:   {'Available' if deps['librosa'] else 'NOT FOUND (BPM auto-detect disabled)'}")
    print(f"  Framework: {'Available' if deps['framework'] else 'NOT FOUND'}")

    print("\nAnalyzer ready.")
