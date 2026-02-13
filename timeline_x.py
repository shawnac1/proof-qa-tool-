"""
Timeline X - Ultimate AI-Powered Timeline Creator
By Aerial Canvas

The AI analyzes raw footage, makes editorial decisions, and outputs
structured timelines with cuts, transitions, music sync, pacing, and flow.
The editor can refine from there - the heavy lift of assembly is done.

Exports to:
- DaVinci Resolve XML
- Final Cut Pro FCPXML
- Adobe Premiere Pro XML

Built on the editorial principles of Murch, Schoonmaker, Coates, Menke, and the masters.
"""

import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import uuid
import math

# Import the comprehensive framework
try:
    from timeline_x_framework import (
        ShotType as FrameworkShotType,
        CameraMovement, ContentCategory, TechnicalQuality,
        ClipAnalysis, AudioAnalysis, ContentMap, BeatMap,
        SEQUENCING_RULES, MUSIC_SYNC_RULES, ENERGY_MATCHING_RULES,
        VO_ALIGNMENT_RULES, TRANSITION_RULES, FORMAT_TEMPLATES,
        AUTO_COLOR_SETTINGS, AUTO_AUDIO_SETTINGS, QUALITY_CHECKS,
        get_format_template, get_transition_recommendation,
        calculate_shot_duration, should_cut_on_beat, evaluate_quality
    )
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False

# Import the media analyzer
try:
    from timeline_x_analyzer import (
        FFProbeAnalyzer, BPMAnalyzer, ClipAnalyzer,
        check_dependencies, format_duration, format_timecode
    )
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

# ============================================================================
# VERSION & METADATA
# ============================================================================

TIMELINE_X_VERSION = "0.3.0"
TIMELINE_X_NAME = "Timeline X"

# ============================================================================
# ANALYZER SINGLETON
# ============================================================================

_clip_analyzer = None

def get_clip_analyzer():
    """Get or create the clip analyzer singleton"""
    global _clip_analyzer
    if _clip_analyzer is None and ANALYZER_AVAILABLE:
        _clip_analyzer = ClipAnalyzer()
    return _clip_analyzer

# ============================================================================
# THE EDITORIAL CANON - Masters and Their Principles
# ============================================================================

EDITORIAL_MASTERS = {
    "walter_murch": {
        "name": "Walter Murch",
        "works": ["Apocalypse Now", "The Godfather Part II & III", "The English Patient", "The Conversation"],
        "philosophy": "Rule of Six - prioritize emotion over technical perfection",
        "key_text": "In the Blink of an Eye",
        "principles": [
            "Emotion trumps all other considerations",
            "Story advancement is second priority",
            "Rhythm and musical flow third",
            "Eye trace continuity fourth",
            "2D screen plane fifth",
            "3D spatial continuity last - sacrifice this first if needed"
        ]
    },
    "thelma_schoonmaker": {
        "name": "Thelma Schoonmaker",
        "works": ["Raging Bull", "Goodfellas", "The Wolf of Wall Street", "The Departed", "The Irishman"],
        "philosophy": "Energy and rhythm through performance",
        "principles": [
            "Cut on action and energy",
            "Cut on internal rhythm of performance",
            "Sometimes the most powerful edit is the one you don't make",
            "Letting a shot breathe creates tension a cut would release"
        ]
    },
    "anne_v_coates": {
        "name": "Anne V. Coates",
        "works": ["Lawrence of Arabia", "The Elephant Man", "Out of Sight", "Erin Brockovich"],
        "philosophy": "Transitions carry metaphorical weight",
        "principles": [
            "A single cut can carry enormous meaning",
            "Match cuts are statements about transformation",
            "Transitions should carry meaning, not just be functional"
        ]
    },
    "sally_menke": {
        "name": "Sally Menke",
        "works": ["Reservoir Dogs", "Pulp Fiction", "Kill Bill", "Inglourious Basterds"],
        "philosophy": "Tension through deliberate pacing and withheld information",
        "principles": [
            "Let tension build through deliberate pacing",
            "Strategic withholding of the cut builds unbearable tension",
            "Hold the wide when audience wants the close-up",
            "Pacing is about control of information and anticipation"
        ]
    },
    "tom_cross": {
        "name": "Tom Cross",
        "works": ["Whiplash", "La La Land", "First Man"],
        "philosophy": "Rhythm as editing - cuts measured in frames create physical experience",
        "principles": [
            "Cut to musical tempo for visceral impact",
            "Frame-perfect timing creates tension and exhilaration",
            "Rhythmic precision especially critical with music"
        ]
    },
    "margaret_sixel": {
        "name": "Margaret Sixel",
        "works": ["Mad Max Fury Road"],
        "philosophy": "Clarity in chaos through center-framing",
        "principles": [
            "Keep primary visual information center-frame across cuts",
            "Maintain strict spatial orientation even at 2-3 cuts per second",
            "Fast pacing must maintain spatial clarity"
        ]
    },
    "kirk_baxter_angus_wall": {
        "name": "Kirk Baxter & Angus Wall",
        "works": ["The Social Network", "Gone Girl", "The Girl with the Dragon Tattoo", "Zodiac"],
        "philosophy": "Invisible precision - cuts feel inevitable",
        "principles": [
            "Every cut motivated by performance, dialogue, or visual logic",
            "Cuts should feel inevitable, not arbitrary",
            "Seamless cross-cutting between multiple timelines"
        ]
    },
    "hank_corwin": {
        "name": "Hank Corwin",
        "works": ["The Big Short", "Natural Born Killers", "The Tree of Life", "Don't Look Up"],
        "philosophy": "Aggressive, confrontational editing as valid creative choice",
        "principles": [
            "Breaking rules can serve content",
            "Smash cuts, flash frames, mixed formats for energy",
            "Unconventional editing makes complex information engaging"
        ]
    }
}

# ============================================================================
# WALTER MURCH'S RULE OF SIX - The Core Evaluation Framework
# ============================================================================

class MurchPriority(Enum):
    """The Rule of Six - in order of importance (sacrifice from bottom up)"""
    EMOTION = 1           # Does the cut FEEL right?
    STORY = 2             # Does the cut advance the narrative?
    RHYTHM = 3            # Is it at the right moment in the rhythmic flow?
    EYE_TRACE = 4         # Does viewer's eye find the new focal point naturally?
    TWO_D_PLANE = 5       # Does it maintain visual continuity on flat screen?
    THREE_D_SPACE = 6     # Does it maintain physical spatial continuity?


# ============================================================================
# SHOT TYPES & CLASSIFICATIONS
# ============================================================================

class ShotSize(Enum):
    """Standard shot size classifications"""
    EXTREME_WIDE = "EWS"      # Vast landscape, tiny subject
    WIDE = "WS"               # Full environment, full body
    MEDIUM_WIDE = "MWS"       # Knees up, environment context
    MEDIUM = "MS"             # Waist up
    MEDIUM_CLOSE = "MCU"      # Chest up
    CLOSE_UP = "CU"           # Face fills frame
    EXTREME_CLOSE = "ECU"     # Eyes, detail
    INSERT = "INSERT"         # Object detail


class ShotType(Enum):
    """Functional shot classifications"""
    ESTABLISHING = "establishing"
    MASTER = "master"
    COVERAGE = "coverage"
    REACTION = "reaction"
    INSERT = "insert"
    CUTAWAY = "cutaway"
    POV = "pov"
    OVER_SHOULDER = "ots"
    TWO_SHOT = "two_shot"
    GROUP = "group"
    AERIAL = "aerial"
    TRACKING = "tracking"
    STATIC = "static"
    HANDHELD = "handheld"
    GIMBAL = "gimbal"


class TransitionType(Enum):
    """Available transition types"""
    CUT = "cut"                    # Hard cut - default, invisible
    DISSOLVE = "dissolve"          # Cross-dissolve - time passage, mood
    FADE_IN = "fade_in"            # From black - major section start
    FADE_OUT = "fade_out"          # To black - major section end
    WIPE = "wipe"                  # Directional push - energetic, retro
    WHIP_PAN = "whip_pan"          # Motion blur transition - high energy
    MATCH_CUT = "match_cut"        # Visual similarity link - sophisticated
    SMASH_CUT = "smash_cut"        # Jarring contrast - intentional shock
    J_CUT = "j_cut"                # Audio leads video
    L_CUT = "l_cut"                # Audio trails video


# ============================================================================
# CONTENT FORMAT DEFINITIONS
# ============================================================================

class ContentFormat(Enum):
    """Supported content formats with their specific rules"""
    SHORT_FILM = "short_film"
    DOCUMENTARY = "documentary"
    BRAND_FILM = "brand_film"
    COMMERCIAL = "commercial"
    MUSIC_VIDEO = "music_video"
    TESTIMONIAL = "testimonial"
    SOCIAL_REEL = "social_reel"
    REAL_ESTATE = "real_estate"
    EVENT_RECAP = "event_recap"
    CORPORATE = "corporate"
    WEDDING = "wedding"
    TUTORIAL = "tutorial"
    VLOG = "vlog"


# Format-specific timing and structure rules
FORMAT_RULES = {
    ContentFormat.SHORT_FILM: {
        "duration_range": (300, 1200),  # 5-20 minutes in seconds
        "avg_shot_duration": (3, 8),
        "structure": {
            "setup": (0.0, 0.20),        # First 15-20%
            "inciting_incident": 0.18,    # By 18% mark
            "rising_action": (0.20, 0.70),
            "climax": (0.70, 0.80),
            "resolution": (0.80, 1.0)
        },
        "beat_spacing": (5, 15),  # seconds between beats
        "hook_duration": 5,
        "principles": ["Enter scenes late, exit early", "Vary pacing by scene type"]
    },
    ContentFormat.DOCUMENTARY: {
        "duration_range": (1200, 5400),  # 20-90 minutes
        "avg_shot_duration": (5, 15),
        "structure": {
            "central_question": (0.0, 0.05),  # Establish in first 2-3 min
            "exploration": (0.05, 0.70),
            "climax_revelation": (0.70, 0.85),
            "reflection": (0.85, 1.0)
        },
        "beat_spacing": (15, 30),
        "hook_duration": 10,
        "max_interview_hold": 15,  # Never hold on talking head > 15 sec without b-roll
        "principles": ["Intercut interviews with b-roll", "Structure is discovered, not imposed"]
    },
    ContentFormat.BRAND_FILM: {
        "duration_range": (60, 300),  # 1-5 minutes
        "avg_shot_duration": (2, 5),
        "structure": {
            "brand_establish": (0.0, 0.05),  # Brand visible in first 5 sec
            "value_build": (0.05, 0.80),
            "cta": (0.80, 1.0)
        },
        "beat_spacing": (3, 8),
        "hook_duration": 3,
        "principles": ["Music drives pacing", "Each shot serves brand values", "CTA must feel earned"]
    },
    ContentFormat.COMMERCIAL: {
        "duration_range": (15, 60),
        "avg_shot_duration": (1.5, 3),
        "structure": {
            "hook": (0.0, 0.10),
            "problem_desire": (0.0, 0.20),
            "solution": (0.20, 0.65),
            "brand_payoff": (0.65, 1.0)
        },
        "beat_spacing": (2, 5),
        "hook_duration": 1,  # Frame one must be compelling
        "principles": ["Every frame counts", "No wasted shots", "Immediate hook"]
    },
    ContentFormat.MUSIC_VIDEO: {
        "duration_range": (180, 300),  # 3-5 minutes
        "avg_shot_duration": (1, 4),
        "structure": {
            "intro": (0.0, 0.10),
            "verse_1": (0.10, 0.25),
            "chorus_1": (0.25, 0.40),
            "verse_2": (0.40, 0.55),
            "chorus_2": (0.55, 0.70),
            "bridge": (0.70, 0.80),
            "final_chorus": (0.80, 0.95),
            "outro": (0.95, 1.0)
        },
        "beat_spacing": (2, 4),
        "hook_duration": 3,
        "principles": ["Lock cuts to musical structure", "Chorus = faster cutting", "Bridge = visual shift"]
    },
    ContentFormat.TESTIMONIAL: {
        "duration_range": (60, 180),  # 1-3 minutes
        "avg_shot_duration": (3, 7),
        "structure": {
            "compelling_open": (0.0, 0.05),  # Strong statement immediately
            "before_state": (0.05, 0.30),
            "transformation": (0.30, 0.70),
            "after_state": (0.70, 1.0)
        },
        "beat_spacing": (5, 10),
        "hook_duration": 5,
        "principles": ["Cut filler words", "B-roll covers every edit", "No visible jump cuts"]
    },
    ContentFormat.SOCIAL_REEL: {
        "duration_range": (15, 60),
        "avg_shot_duration": (1, 2),
        "structure": {
            "scroll_stop": (0.0, 0.03),  # First frame matters
            "text_hook": (0.0, 0.10),    # Text in first 1-2 sec
            "value_delivery": (0.10, 0.85),
            "payoff": (0.85, 1.0)
        },
        "beat_spacing": (1, 3),
        "hook_duration": 1,
        "principles": ["First frame stops scroll", "Relentless pacing", "Front-load value"]
    },
    ContentFormat.REAL_ESTATE: {
        "duration_range": (30, 90),
        "avg_shot_duration": (3, 6),
        "structure": {
            "hero_shot": (0.0, 0.08),       # Best exterior or reveal
            "neighborhood": (0.08, 0.15),    # Context (drone/street)
            "entry_living": (0.15, 0.35),    # Entry, main living areas
            "kitchen": (0.35, 0.50),         # Kitchen (hero feature)
            "primary_suite": (0.50, 0.65),   # Primary bedroom/bath
            "additional": (0.65, 0.80),      # Other bedrooms
            "outdoor": (0.80, 0.92),         # Outdoor spaces
            "closing": (0.92, 1.0)           # Best drone/twilight + contact
        },
        "beat_spacing": (3, 7),
        "hook_duration": 3,
        "principles": ["Logical tour flow", "Hero features get more time", "Luxury = slower, Standard = energetic"]
    },
    ContentFormat.EVENT_RECAP: {
        "duration_range": (60, 300),
        "avg_shot_duration": (2, 4),
        "structure": {
            "energy_open": (0.0, 0.10),
            "context_setup": (0.10, 0.20),
            "highlights": (0.20, 0.85),
            "emotional_close": (0.85, 1.0)
        },
        "beat_spacing": (3, 6),
        "hook_duration": 3,
        "principles": ["Best moments throughout", "Build energy", "Emotional peaks and valleys"]
    },
    ContentFormat.WEDDING: {
        "duration_range": (180, 600),  # 3-10 minutes for highlight
        "avg_shot_duration": (3, 6),
        "structure": {
            "prep_anticipation": (0.0, 0.20),
            "ceremony": (0.20, 0.50),
            "celebration": (0.50, 0.85),
            "emotional_close": (0.85, 1.0)
        },
        "beat_spacing": (4, 10),
        "hook_duration": 5,
        "principles": ["Emotion over perfection", "Key moments breathe", "Music drives feeling"]
    }
}


# ============================================================================
# PACING & RHYTHM ANALYSIS
# ============================================================================

@dataclass
class PacingCurve:
    """Represents the pacing over the duration of a piece"""
    timestamps: List[float] = field(default_factory=list)
    shot_durations: List[float] = field(default_factory=list)
    energy_levels: List[float] = field(default_factory=list)  # 0-1 scale

    def get_average_duration(self) -> float:
        if not self.shot_durations:
            return 0.0
        return sum(self.shot_durations) / len(self.shot_durations)

    def get_rhythm_variance(self) -> float:
        """Higher variance = more dynamic rhythm, lower = flatline"""
        if len(self.shot_durations) < 2:
            return 0.0
        mean = self.get_average_duration()
        variance = sum((d - mean) ** 2 for d in self.shot_durations) / len(self.shot_durations)
        return math.sqrt(variance)

    def is_flatline(self, threshold: float = 0.5) -> bool:
        """Check if rhythm lacks variation"""
        return self.get_rhythm_variance() < threshold


@dataclass
class Beat:
    """A moment where something shifts in the edit"""
    timestamp: float
    beat_type: str  # "information", "emotional", "visual", "music", "revelation"
    intensity: float  # 0-1
    description: str = ""


# ============================================================================
# SHOT & CLIP DATA STRUCTURES
# ============================================================================

@dataclass
class Clip:
    """Represents a single clip/shot in the timeline"""
    id: str
    file_path: str
    filename: str

    # Timing
    duration: float  # In seconds
    in_point: float  # Source in point
    out_point: float  # Source out point
    timeline_start: float = 0.0  # Position on timeline

    # Classification
    shot_size: Optional[ShotSize] = None
    shot_type: Optional[ShotType] = None

    # Content analysis
    detected_scene: str = ""  # "kitchen", "exterior", "interview", etc.
    detected_objects: List[str] = field(default_factory=list)
    has_speech: bool = False
    transcript: str = ""
    dominant_colors: List[str] = field(default_factory=list)

    # Audio
    has_audio: bool = True
    audio_type: str = ""  # "dialogue", "music", "ambient", "mixed"
    audio_level: float = 0.0  # dB

    # Emotional/narrative markers
    emotional_valence: float = 0.0  # -1 (negative) to 1 (positive)
    energy_level: float = 0.5  # 0 (calm) to 1 (intense)
    narrative_importance: float = 0.5  # 0-1

    # Technical
    resolution: Tuple[int, int] = (1920, 1080)
    frame_rate: float = 24.0
    codec: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Transition:
    """Represents a transition between clips"""
    transition_type: TransitionType = TransitionType.CUT
    duration: float = 0.0  # In frames or seconds

    # For J/L cuts
    audio_offset: float = 0.0

    # Murch evaluation scores (0-1)
    emotion_score: float = 0.5
    story_score: float = 0.5
    rhythm_score: float = 0.5
    eye_trace_score: float = 0.5
    two_d_score: float = 0.5
    three_d_score: float = 0.5


@dataclass
class Track:
    """Represents a track in the timeline (video or audio)"""
    id: str
    name: str
    track_type: str  # "video", "audio", "music", "vo", "sfx"
    track_index: int
    clips: List[Clip] = field(default_factory=list)
    is_muted: bool = False
    is_locked: bool = False


# ============================================================================
# TIMELINE STRUCTURE
# ============================================================================

@dataclass
class TimelineStructure:
    """The narrative structure of the timeline"""
    format: ContentFormat
    total_duration: float

    # Three-act structure
    act_1_end: float = 0.0      # Setup ends
    act_2_end: float = 0.0      # Confrontation ends
    climax_position: float = 0.0

    # Key moments
    hook_end: float = 0.0
    inciting_incident: float = 0.0
    midpoint: float = 0.0

    # Beats
    beats: List[Beat] = field(default_factory=list)

    # Pacing
    pacing_curve: Optional[PacingCurve] = None


@dataclass
class Timeline:
    """The complete timeline structure"""
    id: str
    name: str
    created_at: datetime

    # Settings
    frame_rate: float = 24.0
    resolution: Tuple[int, int] = (1920, 1080)

    # Content
    format: ContentFormat = ContentFormat.BRAND_FILM
    video_tracks: List[Track] = field(default_factory=list)
    audio_tracks: List[Track] = field(default_factory=list)

    # Structure
    structure: Optional[TimelineStructure] = None

    # Analysis scores
    structure_score: float = 0.0
    hook_score: float = 0.0
    pacing_score: float = 0.0
    rhythm_score: float = 0.0
    flow_score: float = 0.0
    emotional_arc_score: float = 0.0
    music_sync_score: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def get_total_duration(self) -> float:
        """Calculate total timeline duration"""
        max_duration = 0.0
        for track in self.video_tracks:
            for clip in track.clips:
                end = clip.timeline_start + clip.duration
                max_duration = max(max_duration, end)
        return max_duration

    def get_all_clips(self) -> List[Clip]:
        """Get all clips from all video tracks"""
        clips = []
        for track in self.video_tracks:
            clips.extend(track.clips)
        return sorted(clips, key=lambda c: c.timeline_start)


# ============================================================================
# STORY ASSEMBLY ENGINE
# ============================================================================

class StoryAssembler:
    """
    The brain of Timeline X - assembles footage into narrative structure
    based on editorial principles.
    """

    def __init__(self, content_format: ContentFormat):
        self.format = content_format
        self.rules = FORMAT_RULES.get(content_format, FORMAT_RULES[ContentFormat.BRAND_FILM])

    def analyze_clips(self, clips: List[Clip]) -> Dict[str, List[Clip]]:
        """
        Analyze and categorize clips by their narrative function.
        Returns clips grouped by: hero, establishing, interview, broll, action, reaction, etc.
        """
        categorized = {
            "hero": [],           # Best/most impressive shots
            "establishing": [],   # Wide shots that set location
            "interview": [],      # Talking heads
            "broll": [],          # Supporting visuals
            "action": [],         # Movement, activity
            "reaction": [],       # Response shots
            "detail": [],         # Insert/detail shots
            "transition": [],     # Shots good for transitions
        }

        for clip in clips:
            # Categorize based on shot type and content
            if clip.shot_type == ShotType.ESTABLISHING or clip.shot_size in [ShotSize.EXTREME_WIDE, ShotSize.WIDE]:
                categorized["establishing"].append(clip)

            if clip.has_speech and clip.shot_size in [ShotSize.MEDIUM, ShotSize.MEDIUM_CLOSE, ShotSize.CLOSE_UP]:
                categorized["interview"].append(clip)

            if clip.shot_type == ShotType.INSERT or clip.shot_size == ShotSize.EXTREME_CLOSE:
                categorized["detail"].append(clip)

            if clip.energy_level > 0.7:
                categorized["action"].append(clip)

            if clip.shot_type == ShotType.REACTION:
                categorized["reaction"].append(clip)

            # Hero shots: high energy OR high narrative importance
            if clip.narrative_importance > 0.8 or (clip.energy_level > 0.6 and clip.shot_size in [ShotSize.WIDE, ShotSize.MEDIUM_WIDE]):
                categorized["hero"].append(clip)

            # Everything else is b-roll
            if clip not in categorized["interview"] and clip not in categorized["detail"]:
                categorized["broll"].append(clip)

        return categorized

    def build_three_act_structure(self, clips: List[Clip], target_duration: float) -> TimelineStructure:
        """
        Build a three-act structure for the given clips.
        """
        structure = TimelineStructure(
            format=self.format,
            total_duration=target_duration
        )

        # Get structure percentages from format rules
        format_structure = self.rules.get("structure", {})

        # Calculate act breaks based on format
        if "setup" in format_structure:
            setup_range = format_structure["setup"]
            structure.act_1_end = target_duration * setup_range[1]
        else:
            structure.act_1_end = target_duration * 0.25

        if "climax" in format_structure:
            climax_range = format_structure["climax"]
            structure.climax_position = target_duration * ((climax_range[0] + climax_range[1]) / 2)
            structure.act_2_end = target_duration * climax_range[0]
        else:
            structure.climax_position = target_duration * 0.75
            structure.act_2_end = target_duration * 0.75

        # Hook duration
        structure.hook_end = self.rules.get("hook_duration", 5)

        # Midpoint
        structure.midpoint = target_duration * 0.5

        return structure

    def calculate_ideal_pacing(self, structure: TimelineStructure) -> PacingCurve:
        """
        Calculate the ideal pacing curve for the structure.
        """
        curve = PacingCurve()

        # Get shot duration range from format rules
        min_dur, max_dur = self.rules.get("avg_shot_duration", (2, 5))
        avg_dur = (min_dur + max_dur) / 2

        # Sample points throughout the timeline
        num_samples = 20
        for i in range(num_samples):
            timestamp = (i / num_samples) * structure.total_duration
            curve.timestamps.append(timestamp)

            # Calculate energy based on position in structure
            position_pct = timestamp / structure.total_duration

            # Energy curve: builds to climax, then releases
            if position_pct < 0.1:  # Hook - high energy
                energy = 0.8
                shot_dur = min_dur
            elif position_pct < 0.25:  # Setup - moderate
                energy = 0.5
                shot_dur = avg_dur
            elif position_pct < 0.5:  # Rising action - building
                energy = 0.5 + (position_pct - 0.25) * 1.2
                shot_dur = avg_dur - (position_pct - 0.25) * (avg_dur - min_dur)
            elif position_pct < 0.75:  # Approaching climax - high
                energy = 0.8 + (position_pct - 0.5) * 0.4
                shot_dur = min_dur
            elif position_pct < 0.85:  # Climax - peak
                energy = 1.0
                shot_dur = min_dur * 0.8
            else:  # Resolution - release
                energy = 1.0 - (position_pct - 0.85) * 4
                shot_dur = avg_dur + (position_pct - 0.85) * (max_dur - avg_dur) * 2

            curve.energy_levels.append(min(1.0, max(0.0, energy)))
            curve.shot_durations.append(max(min_dur * 0.5, min(max_dur * 1.5, shot_dur)))

        return curve

    def assemble_timeline(self, clips: List[Clip], target_duration: Optional[float] = None) -> Timeline:
        """
        Assemble clips into a complete timeline following editorial principles.
        """
        # Analyze and categorize clips
        categorized = self.analyze_clips(clips)

        # Calculate target duration if not specified
        if target_duration is None:
            total_footage = sum(c.duration for c in clips)
            # Typical edit uses 10-20% of total footage
            target_duration = total_footage * 0.15
            # Clamp to format range
            min_dur, max_dur = self.rules.get("duration_range", (60, 300))
            target_duration = max(min_dur, min(max_dur, target_duration))

        # Build structure
        structure = self.build_three_act_structure(clips, target_duration)
        structure.pacing_curve = self.calculate_ideal_pacing(structure)

        # Create timeline
        timeline = Timeline(
            id=str(uuid.uuid4()),
            name=f"Timeline X - {self.format.value.replace('_', ' ').title()}",
            created_at=datetime.now(),
            format=self.format,
            structure=structure
        )

        # Create main video track
        main_track = Track(
            id="V1",
            name="Video 1",
            track_type="video",
            track_index=1
        )

        # Assemble clips according to structure
        current_time = 0.0

        # Act 1: Setup
        # Start with hero/establishing shot
        if categorized["hero"]:
            hero = categorized["hero"][0]
            hero.timeline_start = current_time
            hero.duration = min(hero.duration, structure.hook_end)
            main_track.clips.append(hero)
            current_time += hero.duration

        # Add establishing shots
        for clip in categorized["establishing"][:2]:
            if current_time >= structure.act_1_end:
                break
            clip.timeline_start = current_time
            clip.duration = min(clip.duration, 4.0)  # Keep establishing shots brief
            main_track.clips.append(clip)
            current_time += clip.duration

        # Act 2: Main content
        # Interleave interview with b-roll
        interview_clips = categorized["interview"]
        broll_clips = categorized["broll"]

        interview_idx = 0
        broll_idx = 0

        while current_time < structure.act_2_end:
            # Add interview segment
            if interview_idx < len(interview_clips):
                clip = interview_clips[interview_idx]
                clip.timeline_start = current_time
                # Limit interview hold time per format rules
                max_hold = self.rules.get("max_interview_hold", 10)
                clip.duration = min(clip.duration, max_hold)
                main_track.clips.append(clip)
                current_time += clip.duration
                interview_idx += 1

            # Add b-roll
            if broll_idx < len(broll_clips):
                clip = broll_clips[broll_idx]
                clip.timeline_start = current_time
                # B-roll duration based on pacing curve
                clip.duration = min(clip.duration, 5.0)
                main_track.clips.append(clip)
                current_time += clip.duration
                broll_idx += 1

            # Safety check
            if interview_idx >= len(interview_clips) and broll_idx >= len(broll_clips):
                break

        # Act 3: Resolution
        # End with strong visuals
        remaining_heroes = [c for c in categorized["hero"] if c not in main_track.clips]
        for clip in remaining_heroes[:2]:
            if current_time >= target_duration:
                break
            clip.timeline_start = current_time
            main_track.clips.append(clip)
            current_time += clip.duration

        timeline.video_tracks.append(main_track)

        return timeline


# ============================================================================
# XML EXPORT - DAVINCI RESOLVE
# ============================================================================

class DaVinciResolveExporter:
    """Export timeline to DaVinci Resolve XML format"""

    def __init__(self, timeline: Timeline):
        self.timeline = timeline

    def export(self, output_path: str) -> str:
        """Export timeline to DaVinci Resolve XML"""

        root = ET.Element("xmeml", version="5")

        # Project
        project = ET.SubElement(root, "project")
        ET.SubElement(project, "name").text = self.timeline.name

        # Sequence
        sequence = ET.SubElement(root, "sequence")
        ET.SubElement(sequence, "name").text = self.timeline.name
        ET.SubElement(sequence, "duration").text = str(int(self.timeline.get_total_duration() * self.timeline.frame_rate))

        # Rate
        rate = ET.SubElement(sequence, "rate")
        ET.SubElement(rate, "timebase").text = str(int(self.timeline.frame_rate))
        ET.SubElement(rate, "ntsc").text = "FALSE"

        # Timecode
        timecode = ET.SubElement(sequence, "timecode")
        ET.SubElement(timecode, "string").text = "00:00:00:00"
        tc_rate = ET.SubElement(timecode, "rate")
        ET.SubElement(tc_rate, "timebase").text = str(int(self.timeline.frame_rate))

        # Media
        media = ET.SubElement(sequence, "media")
        video = ET.SubElement(media, "video")

        # Video format
        v_format = ET.SubElement(video, "format")
        sample_char = ET.SubElement(v_format, "samplecharacteristics")
        ET.SubElement(sample_char, "width").text = str(self.timeline.resolution[0])
        ET.SubElement(sample_char, "height").text = str(self.timeline.resolution[1])

        # Video tracks
        for track in self.timeline.video_tracks:
            v_track = ET.SubElement(video, "track")

            for clip in track.clips:
                clip_item = ET.SubElement(v_track, "clipitem", id=clip.id)
                ET.SubElement(clip_item, "name").text = clip.filename
                ET.SubElement(clip_item, "duration").text = str(int(clip.duration * self.timeline.frame_rate))
                ET.SubElement(clip_item, "start").text = str(int(clip.timeline_start * self.timeline.frame_rate))
                ET.SubElement(clip_item, "end").text = str(int((clip.timeline_start + clip.duration) * self.timeline.frame_rate))
                ET.SubElement(clip_item, "in").text = str(int(clip.in_point * self.timeline.frame_rate))
                ET.SubElement(clip_item, "out").text = str(int(clip.out_point * self.timeline.frame_rate))

                # File reference
                file_elem = ET.SubElement(clip_item, "file", id=f"file-{clip.id}")
                ET.SubElement(file_elem, "name").text = clip.filename
                ET.SubElement(file_elem, "pathurl").text = f"file://localhost{clip.file_path}"

        # Pretty print
        xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

        return output_path


# ============================================================================
# XML EXPORT - FINAL CUT PRO (FCPXML)
# ============================================================================

class FinalCutProExporter:
    """Export timeline to Final Cut Pro FCPXML format"""

    def __init__(self, timeline: Timeline):
        self.timeline = timeline

    def _frames_to_rational(self, frames: int) -> str:
        """Convert frames to rational time format (e.g., '1001/30000s')"""
        # For 24fps: 1 frame = 1001/24000s
        numerator = int(frames * 1001)
        denominator = int(self.timeline.frame_rate * 1000)
        return f"{numerator}/{denominator}s"

    def _seconds_to_rational(self, seconds: float) -> str:
        """Convert seconds to rational time format"""
        frames = int(seconds * self.timeline.frame_rate)
        return self._frames_to_rational(frames)

    def export(self, output_path: str) -> str:
        """Export timeline to FCPXML"""

        root = ET.Element("fcpxml", version="1.9")

        # Resources
        resources = ET.SubElement(root, "resources")

        # Format resource
        format_id = "r1"
        format_elem = ET.SubElement(resources, "format",
            id=format_id,
            name=f"{self.timeline.resolution[0]}x{self.timeline.resolution[1]}p{int(self.timeline.frame_rate)}",
            frameDuration=self._frames_to_rational(1),
            width=str(self.timeline.resolution[0]),
            height=str(self.timeline.resolution[1])
        )

        # Asset resources for each clip
        asset_map = {}
        for idx, track in enumerate(self.timeline.video_tracks):
            for clip in track.clips:
                asset_id = f"r{len(asset_map) + 2}"
                asset_map[clip.id] = asset_id

                asset = ET.SubElement(resources, "asset",
                    id=asset_id,
                    name=clip.filename,
                    src=f"file://{clip.file_path}",
                    duration=self._seconds_to_rational(clip.out_point - clip.in_point),
                    hasVideo="1",
                    hasAudio="1" if clip.has_audio else "0"
                )

        # Library
        library = ET.SubElement(root, "library")

        # Event
        event = ET.SubElement(library, "event", name="Timeline X Export")

        # Project
        project = ET.SubElement(event, "project", name=self.timeline.name)

        # Sequence
        sequence = ET.SubElement(project, "sequence",
            format=format_id,
            duration=self._seconds_to_rational(self.timeline.get_total_duration()),
            tcStart="0s",
            tcFormat="NDF"
        )

        # Spine (main timeline)
        spine = ET.SubElement(sequence, "spine")

        # Add clips to spine
        for track in self.timeline.video_tracks:
            for clip in track.clips:
                asset_id = asset_map.get(clip.id, "r2")

                # Calculate gap if needed
                # For now, just add clips sequentially

                asset_clip = ET.SubElement(spine, "asset-clip",
                    ref=asset_id,
                    name=clip.filename,
                    offset=self._seconds_to_rational(clip.timeline_start),
                    duration=self._seconds_to_rational(clip.duration),
                    start=self._seconds_to_rational(clip.in_point),
                    tcFormat="NDF"
                )

        # Pretty print
        xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

        return output_path


# ============================================================================
# XML EXPORT - ADOBE PREMIERE PRO
# ============================================================================

class PremiereProExporter:
    """Export timeline to Adobe Premiere Pro XML format"""

    def __init__(self, timeline: Timeline):
        self.timeline = timeline

    def export(self, output_path: str) -> str:
        """Export timeline to Premiere Pro XML"""

        root = ET.Element("xmeml", version="4")

        # Sequence
        sequence = ET.SubElement(root, "sequence")
        ET.SubElement(sequence, "name").text = self.timeline.name

        total_frames = int(self.timeline.get_total_duration() * self.timeline.frame_rate)
        ET.SubElement(sequence, "duration").text = str(total_frames)

        # Rate
        rate = ET.SubElement(sequence, "rate")
        ET.SubElement(rate, "timebase").text = str(int(self.timeline.frame_rate))
        ET.SubElement(rate, "ntsc").text = "FALSE"

        # Timecode
        timecode = ET.SubElement(sequence, "timecode")
        ET.SubElement(timecode, "string").text = "00:00:00:00"
        ET.SubElement(timecode, "frame").text = "0"
        ET.SubElement(timecode, "displayformat").text = "NDF"
        tc_rate = ET.SubElement(timecode, "rate")
        ET.SubElement(tc_rate, "timebase").text = str(int(self.timeline.frame_rate))

        # Media
        media = ET.SubElement(sequence, "media")

        # Video
        video = ET.SubElement(media, "video")

        # Format
        v_format = ET.SubElement(video, "format")
        sample_char = ET.SubElement(v_format, "samplecharacteristics")
        ET.SubElement(sample_char, "width").text = str(self.timeline.resolution[0])
        ET.SubElement(sample_char, "height").text = str(self.timeline.resolution[1])
        ET.SubElement(sample_char, "pixelaspectratio").text = "square"

        sc_rate = ET.SubElement(sample_char, "rate")
        ET.SubElement(sc_rate, "timebase").text = str(int(self.timeline.frame_rate))

        # Video tracks
        for track in self.timeline.video_tracks:
            v_track = ET.SubElement(video, "track")
            ET.SubElement(v_track, "locked").text = "FALSE"
            ET.SubElement(v_track, "enabled").text = "TRUE"

            for clip in track.clips:
                clip_item = ET.SubElement(v_track, "clipitem", id=clip.id)

                ET.SubElement(clip_item, "name").text = clip.filename
                ET.SubElement(clip_item, "enabled").text = "TRUE"

                clip_duration = int(clip.duration * self.timeline.frame_rate)
                clip_start = int(clip.timeline_start * self.timeline.frame_rate)
                clip_in = int(clip.in_point * self.timeline.frame_rate)
                clip_out = int(clip.out_point * self.timeline.frame_rate)

                ET.SubElement(clip_item, "duration").text = str(clip_duration)
                ET.SubElement(clip_item, "start").text = str(clip_start)
                ET.SubElement(clip_item, "end").text = str(clip_start + clip_duration)
                ET.SubElement(clip_item, "in").text = str(clip_in)
                ET.SubElement(clip_item, "out").text = str(clip_out)

                # Rate
                clip_rate = ET.SubElement(clip_item, "rate")
                ET.SubElement(clip_rate, "timebase").text = str(int(self.timeline.frame_rate))

                # File
                file_elem = ET.SubElement(clip_item, "file", id=f"file-{clip.id}")
                ET.SubElement(file_elem, "name").text = clip.filename
                ET.SubElement(file_elem, "pathurl").text = f"file://localhost{clip.file_path}"

                file_rate = ET.SubElement(file_elem, "rate")
                ET.SubElement(file_rate, "timebase").text = str(int(clip.frame_rate))

                # Media info
                file_media = ET.SubElement(file_elem, "media")
                file_video = ET.SubElement(file_media, "video")
                file_sc = ET.SubElement(file_video, "samplecharacteristics")
                ET.SubElement(file_sc, "width").text = str(clip.resolution[0])
                ET.SubElement(file_sc, "height").text = str(clip.resolution[1])

        # Audio tracks
        audio = ET.SubElement(media, "audio")

        for track in self.timeline.audio_tracks:
            a_track = ET.SubElement(audio, "track")
            ET.SubElement(a_track, "locked").text = "FALSE"
            ET.SubElement(a_track, "enabled").text = "TRUE"

        # Pretty print
        xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

        return output_path


# ============================================================================
# MAIN TIMELINE X CLASS
# ============================================================================

class TimelineX:
    """
    Timeline X - The Ultimate AI-Powered Timeline Creator

    Analyzes footage and generates story-driven timelines for
    DaVinci Resolve, Final Cut Pro, and Adobe Premiere Pro.
    """

    def __init__(self):
        self.clips: List[Clip] = []
        self.timeline: Optional[Timeline] = None
        self.format: ContentFormat = ContentFormat.BRAND_FILM
        self.music_track: Optional[Dict] = None
        self.beat_map: Optional[Dict] = None
        self.analyzer = get_clip_analyzer() if ANALYZER_AVAILABLE else None

    def add_clip(self, clip: Clip):
        """Add a clip to the source pool"""
        self.clips.append(clip)

    def add_clip_from_file(self, file_path: str) -> Optional[Clip]:
        """
        Add a single video file with full analysis.
        Uses FFprobe to get real duration, resolution, frame rate.
        """
        if not os.path.exists(file_path):
            return None

        filename = os.path.basename(file_path)
        clip_id = str(uuid.uuid4())[:8]

        # Try to analyze with FFprobe
        metadata = None
        if self.analyzer and hasattr(self.analyzer, 'ffprobe'):
            metadata = self.analyzer.ffprobe.analyze_video(file_path)

        if metadata:
            clip = Clip(
                id=clip_id,
                file_path=file_path,
                filename=filename,
                duration=metadata["duration"],
                in_point=0.0,
                out_point=metadata["duration"],
                resolution=(metadata["width"], metadata["height"]),
                frame_rate=metadata["frame_rate"],
                codec=metadata["codec"],
                has_audio=metadata.get("has_audio", False),
            )
        else:
            # Fallback - create clip without analysis
            clip = Clip(
                id=clip_id,
                file_path=file_path,
                filename=filename,
                duration=0.0,
                in_point=0.0,
                out_point=0.0
            )

        self.clips.append(clip)
        return clip

    def add_clips_from_folder(self, folder_path: str, analyze: bool = True) -> int:
        """
        Scan a folder and add all video clips.
        If analyze=True, uses FFprobe to get real metadata.
        """
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.prores', '.r3d', '.braw', '.m4v'}
        added_count = 0

        for root, dirs, files in os.walk(folder_path):
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext in video_extensions:
                    file_path = os.path.join(root, filename)

                    if analyze:
                        clip = self.add_clip_from_file(file_path)
                        if clip:
                            added_count += 1
                    else:
                        # Quick add without analysis
                        clip = Clip(
                            id=str(uuid.uuid4())[:8],
                            file_path=file_path,
                            filename=filename,
                            duration=0.0,
                            in_point=0.0,
                            out_point=0.0
                        )
                        self.clips.append(clip)
                        added_count += 1

        return added_count

    def set_music(self, file_path: str, bpm: Optional[float] = None, auto_detect: bool = True) -> bool:
        """
        Set the music track for the timeline.
        If auto_detect=True and bpm is None, attempts to detect BPM.
        """
        if not os.path.exists(file_path):
            return False

        filename = os.path.basename(file_path)

        self.music_track = {
            "file_path": file_path,
            "filename": filename,
            "bpm": bpm or 0.0,
            "duration": 0.0,
            "beat_positions": [],
        }

        # Try to analyze music
        if self.analyzer:
            if auto_detect and bpm is None:
                # Full analysis with BPM detection
                audio_analysis = self.analyzer.analyze_music(file_path)
                if audio_analysis:
                    self.music_track["bpm"] = audio_analysis.bpm
                    self.music_track["duration"] = audio_analysis.duration
                    self.music_track["beat_positions"] = audio_analysis.beat_positions
                    self.beat_map = {
                        "bpm": audio_analysis.bpm,
                        "beats": audio_analysis.beat_positions,
                        "downbeats": audio_analysis.downbeat_positions,
                        "phrases": audio_analysis.phrase_boundaries,
                        "energy": audio_analysis.energy_curve,
                    }
                    return True

            elif bpm is not None and bpm > 0:
                # Generate beat grid from provided BPM
                if hasattr(self.analyzer, 'bpm_analyzer'):
                    # Get duration first
                    audio_data = self.analyzer.bpm_analyzer.analyze_audio(file_path)
                    if audio_data:
                        duration = audio_data["duration"]
                        grid = self.analyzer.bpm_analyzer.generate_beat_grid(bpm, duration)
                        if grid:
                            self.music_track["bpm"] = bpm
                            self.music_track["duration"] = duration
                            self.music_track["beat_positions"] = grid["beat_positions"]
                            self.beat_map = {
                                "bpm": bpm,
                                "beats": grid["beat_positions"],
                                "downbeats": grid["downbeats"],
                                "phrases": grid["phrase_boundaries"],
                            }
                            return True

        return False

    def get_total_footage_duration(self) -> float:
        """Get total duration of all source clips"""
        return sum(c.duration for c in self.clips if c.duration > 0)

    def get_clips_summary(self) -> Dict[str, Any]:
        """Get summary of loaded clips"""
        total_duration = self.get_total_footage_duration()
        analyzed = sum(1 for c in self.clips if c.duration > 0)

        return {
            "total_clips": len(self.clips),
            "analyzed_clips": analyzed,
            "total_duration": total_duration,
            "avg_duration": total_duration / len(self.clips) if self.clips else 0,
            "has_music": self.music_track is not None,
            "music_bpm": self.music_track["bpm"] if self.music_track else 0,
        }

    def set_format(self, format: ContentFormat):
        """Set the content format for timeline generation"""
        self.format = format

    def generate_timeline(self, target_duration: Optional[float] = None) -> Timeline:
        """
        Generate a timeline from the loaded clips.

        Uses the StoryAssembler to analyze clips and build a narrative structure
        following the editorial principles of the masters.
        """
        if not self.clips:
            raise ValueError("No clips loaded. Use add_clip() or add_clips_from_folder() first.")

        assembler = StoryAssembler(self.format)
        self.timeline = assembler.assemble_timeline(self.clips, target_duration)

        return self.timeline

    def export_davinci(self, output_path: str) -> str:
        """Export timeline to DaVinci Resolve XML"""
        if not self.timeline:
            raise ValueError("No timeline generated. Call generate_timeline() first.")

        exporter = DaVinciResolveExporter(self.timeline)
        return exporter.export(output_path)

    def export_fcpxml(self, output_path: str) -> str:
        """Export timeline to Final Cut Pro FCPXML"""
        if not self.timeline:
            raise ValueError("No timeline generated. Call generate_timeline() first.")

        exporter = FinalCutProExporter(self.timeline)
        return exporter.export(output_path)

    def export_premiere(self, output_path: str) -> str:
        """Export timeline to Adobe Premiere Pro XML"""
        if not self.timeline:
            raise ValueError("No timeline generated. Call generate_timeline() first.")

        exporter = PremiereProExporter(self.timeline)
        return exporter.export(output_path)

    def export_all(self, output_folder: str, base_name: str = "timeline_x") -> Dict[str, str]:
        """Export timeline to all three NLE formats"""
        os.makedirs(output_folder, exist_ok=True)

        exports = {}

        # DaVinci Resolve
        davinci_path = os.path.join(output_folder, f"{base_name}_davinci.xml")
        exports["davinci"] = self.export_davinci(davinci_path)

        # Final Cut Pro
        fcpxml_path = os.path.join(output_folder, f"{base_name}.fcpxml")
        exports["fcpxml"] = self.export_fcpxml(fcpxml_path)

        # Premiere Pro
        premiere_path = os.path.join(output_folder, f"{base_name}_premiere.xml")
        exports["premiere"] = self.export_premiere(premiere_path)

        return exports


# ============================================================================
# MAIN ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  {TIMELINE_X_NAME} v{TIMELINE_X_VERSION}")
    print(f"  The Ultimate AI-Powered Timeline Creator")
    print(f"  By Aerial Canvas")
    print(f"{'='*60}\n")

    # Check dependencies
    print("System Status:")
    print(f"  Framework:  {'Ready' if FRAMEWORK_AVAILABLE else 'NOT FOUND'}")
    print(f"  Analyzer:   {'Ready' if ANALYZER_AVAILABLE else 'NOT FOUND'}")

    if ANALYZER_AVAILABLE:
        deps = check_dependencies()
        print(f"  FFprobe:    {'Ready' if deps['ffprobe'] else 'NOT FOUND'}")
        if deps['ffprobe']:
            print(f"              {deps['ffprobe_path']}")
        print(f"  Librosa:    {'Ready (BPM auto-detect enabled)' if deps['librosa'] else 'NOT FOUND (manual BPM only)'}")

    print("\nSupported Formats:")
    for fmt in ContentFormat:
        rules = FORMAT_RULES.get(fmt, {})
        duration = rules.get("duration_range", (0, 0))
        shot_dur = rules.get("avg_shot_duration", (0, 0))
        print(f"  - {fmt.value.replace('_', ' ').title()}")
        print(f"    Duration: {duration[0]//60}-{duration[1]//60} min | Avg shot: {shot_dur[0]}-{shot_dur[1]}s")

    print("\nEditorial Masters Encoded:")
    for key, master in EDITORIAL_MASTERS.items():
        print(f"  - {master['name']}: {master['philosophy']}")

    print("\nExport Formats:")
    print("  - DaVinci Resolve XML")
    print("  - Final Cut Pro FCPXML")
    print("  - Adobe Premiere Pro XML")

    print("\nTimeline X ready.")
