"""
LOCAL FOLDER PROCESSOR
======================

Processes local folders for Proof by Aerial Canvas.
Scans, analyzes, sorts, renames, and generates XML timelines.

Features:
- Folder structure detection (RAW Videos, RAW Photos, etc.)
- Batch video/photo analysis
- Smart file renaming based on content
- XML timeline generation for NLEs
- Works entirely offline with local files

By Aerial Canvas
"""

import os
import shutil
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import re

# Import analyzers
try:
    from timeline_x_analyzer import FFProbeAnalyzer, ClipAnalyzer, check_dependencies
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

# Import Timeline X for XML generation
try:
    from timeline_x import TimelineX, ContentFormat, Clip
    TIMELINE_X_AVAILABLE = True
except ImportError:
    TIMELINE_X_AVAILABLE = False


# ============================================================================
# FOLDER STRUCTURE PATTERNS
# ============================================================================

# Common folder name patterns in Aerial Canvas projects
FOLDER_PATTERNS = {
    "raw_videos": [
        r"raw\s*videos?\s*\(?#*\)?",
        r"raw\s*video\s*clips?",
        r"video\s*raw",
        r"videos?\s*for\s*qa",
    ],
    "raw_photos": [
        r"raw\s*photos?\s*\(?#*\)?",
        r"photos?\s*raw",
        r"interior\s*photos?",
    ],
    "raw_drone_photos": [
        r"raw\s*drone\s*photos?\s*\(?#*\)?",
        r"drone\s*photos?\s*raw",
        r"aerial\s*photos?",
    ],
    "raw_drone_videos": [
        r"raw\s*drone\s*videos?\s*\(?#*\)?",
        r"drone\s*videos?\s*raw",
        r"aerial\s*videos?",
    ],
    "raw_twilight": [
        r"raw\s*twilight\s*\(?#*\)?",
        r"twilight\s*raw",
        r"dusk\s*photos?",
    ],
    "raw_lifestyle": [
        r"raw\s*lifestyle",
        r"lifestyle\s*raw",
        r"lifestyle\s*photos?",
    ],
    "deliverables": [
        r"deliverables?",
        r"final\s*files?",
        r"exports?",
    ],
}

# File extensions
VIDEO_EXTENSIONS = {'.mov', '.mp4', '.avi', '.mkv', '.mxf', '.m4v', '.prores', '.r3d', '.braw'}
PHOTO_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.raw', '.cr2', '.cr3', '.nef', '.arw', '.dng', '.heic'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aiff', '.aac', '.m4a', '.flac'}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DetectedFolder:
    """Represents a detected content folder"""
    path: str
    name: str
    folder_type: str  # "raw_videos", "raw_photos", etc.
    file_count: int
    total_size: int  # bytes
    files: List[str] = field(default_factory=list)


@dataclass
class AnalyzedClip:
    """Represents an analyzed video clip"""
    original_path: str
    original_name: str

    # Analysis results
    duration: float = 0.0
    resolution: Tuple[int, int] = (1920, 1080)
    frame_rate: float = 24.0

    # Content classification
    scene_type: str = "unknown"  # drone, exterior, interior, kitchen, etc.
    shot_type: str = "unknown"   # establishing, walk-through, detail, static
    room_type: str = ""          # For interior shots

    # Quality
    quality_score: int = 80
    issues: List[str] = field(default_factory=list)

    # Sorting
    sort_order: int = 0
    new_name: str = ""


@dataclass
class AnalyzedPhoto:
    """Represents an analyzed photo"""
    original_path: str
    original_name: str

    # Classification
    photo_type: str = "interior"  # drone, exterior, interior, twilight, lifestyle
    room_type: str = "unknown"

    # Quality
    quality_score: int = 80
    issues: List[str] = field(default_factory=list)

    # Sorting
    sort_order: int = 0
    new_name: str = ""


@dataclass
class FolderAnalysisResult:
    """Complete analysis result for a folder"""
    folder_path: str
    folder_type: str

    # Counts
    total_files: int = 0
    analyzed_files: int = 0

    # Results
    clips: List[AnalyzedClip] = field(default_factory=list)
    photos: List[AnalyzedPhoto] = field(default_factory=list)

    # Groupings
    groups: Dict[str, List[str]] = field(default_factory=dict)

    # Timeline
    xml_davinci: str = ""
    xml_fcpxml: str = ""
    xml_premiere: str = ""

    # Summary
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


# ============================================================================
# FOLDER SCANNER
# ============================================================================

class FolderScanner:
    """Scans and detects folder structure"""

    def __init__(self):
        self.detected_folders: List[DetectedFolder] = []

    def scan_path(self, root_path: str) -> Dict[str, Any]:
        """
        Scan a folder path and detect content structure.
        Returns summary of what was found.
        """
        if not os.path.exists(root_path):
            return {"error": f"Path does not exist: {root_path}"}

        if not os.path.isdir(root_path):
            return {"error": f"Path is not a folder: {root_path}"}

        self.detected_folders = []

        # Get immediate contents
        root_name = os.path.basename(root_path)

        # Check if this folder itself is a content folder
        folder_type = self._detect_folder_type(root_name)
        if folder_type:
            files = self._get_media_files(root_path)
            if files:
                self.detected_folders.append(DetectedFolder(
                    path=root_path,
                    name=root_name,
                    folder_type=folder_type,
                    file_count=len(files),
                    total_size=sum(os.path.getsize(f) for f in files),
                    files=files
                ))

        # Scan subfolders
        try:
            for item in os.listdir(root_path):
                item_path = os.path.join(root_path, item)
                if os.path.isdir(item_path):
                    folder_type = self._detect_folder_type(item)
                    files = self._get_media_files(item_path)

                    if files:
                        self.detected_folders.append(DetectedFolder(
                            path=item_path,
                            name=item,
                            folder_type=folder_type or "other",
                            file_count=len(files),
                            total_size=sum(os.path.getsize(f) for f in files),
                            files=files
                        ))
        except PermissionError:
            return {"error": f"Permission denied: {root_path}"}

        # Build summary
        summary = {
            "root_path": root_path,
            "root_name": root_name,
            "folders_detected": len(self.detected_folders),
            "folders": [],
            "total_videos": 0,
            "total_photos": 0,
            "total_size_mb": 0,
        }

        for folder in self.detected_folders:
            video_count = sum(1 for f in folder.files if self._is_video(f))
            photo_count = sum(1 for f in folder.files if self._is_photo(f))

            summary["folders"].append({
                "name": folder.name,
                "type": folder_type_display(folder.folder_type),
                "path": folder.path,
                "videos": video_count,
                "photos": photo_count,
                "size_mb": round(folder.total_size / 1024 / 1024, 1),
            })

            summary["total_videos"] += video_count
            summary["total_photos"] += photo_count
            summary["total_size_mb"] += folder.total_size

        summary["total_size_mb"] = round(summary["total_size_mb"] / 1024 / 1024, 1)

        return summary

    def _detect_folder_type(self, folder_name: str) -> Optional[str]:
        """Detect folder type from name"""
        name_lower = folder_name.lower()

        for folder_type, patterns in FOLDER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower, re.IGNORECASE):
                    return folder_type

        return None

    def _get_media_files(self, folder_path: str) -> List[str]:
        """Get all media files in a folder (non-recursive)"""
        files = []
        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    ext = os.path.splitext(item)[1].lower()
                    if ext in VIDEO_EXTENSIONS or ext in PHOTO_EXTENSIONS:
                        files.append(item_path)
        except PermissionError:
            pass
        return sorted(files)

    def _is_video(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in VIDEO_EXTENSIONS

    def _is_photo(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in PHOTO_EXTENSIONS


def folder_type_display(folder_type: str) -> str:
    """Convert folder type to display name"""
    display_names = {
        "raw_videos": "RAW Videos",
        "raw_photos": "RAW Photos",
        "raw_drone_photos": "Drone Photos",
        "raw_drone_videos": "Drone Videos",
        "raw_twilight": "Twilight",
        "raw_lifestyle": "Lifestyle",
        "deliverables": "Deliverables",
        "other": "Other Media",
    }
    return display_names.get(folder_type, folder_type.replace("_", " ").title())


# ============================================================================
# VIDEO SORTER
# ============================================================================

class VideoSorter:
    """Analyzes and sorts video clips"""

    # Scene type sort order for real estate
    SCENE_ORDER = {
        "drone_establishing": 1,
        "drone_flyover": 2,
        "drone_orbit": 3,
        "drone": 4,
        "exterior_front": 10,
        "exterior": 11,
        "entry": 20,
        "foyer": 21,
        "living_room": 30,
        "family_room": 31,
        "dining_room": 40,
        "kitchen": 50,
        "breakfast_nook": 51,
        "primary_bedroom": 60,
        "primary_bathroom": 61,
        "bedroom": 70,
        "bathroom": 71,
        "office": 80,
        "laundry": 81,
        "garage": 85,
        "backyard": 90,
        "pool": 91,
        "patio": 92,
        "exterior_rear": 95,
        "other": 100,
        "unknown": 999,
    }

    def __init__(self):
        self.ffprobe = FFProbeAnalyzer() if ANALYZER_AVAILABLE else None
        self.clips: List[AnalyzedClip] = []

    def analyze_folder(self, folder_path: str, progress_callback=None) -> FolderAnalysisResult:
        """
        Analyze all video clips in a folder.
        progress_callback(current, total, filename) for progress updates.
        """
        result = FolderAnalysisResult(
            folder_path=folder_path,
            folder_type="raw_videos"
        )

        start_time = datetime.now()

        # Get video files
        video_files = []
        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    ext = os.path.splitext(item)[1].lower()
                    if ext in VIDEO_EXTENSIONS:
                        video_files.append(item_path)
        except Exception as e:
            result.errors.append(f"Error scanning folder: {str(e)}")
            return result

        result.total_files = len(video_files)
        video_files.sort()

        # Analyze each clip
        for idx, file_path in enumerate(video_files):
            filename = os.path.basename(file_path)

            if progress_callback:
                progress_callback(idx + 1, len(video_files), filename)

            clip = self._analyze_clip(file_path)
            self.clips.append(clip)
            result.clips.append(clip)
            result.analyzed_files += 1

        # Sort clips by scene type
        self._sort_clips()

        # Assign new names
        self._assign_names()

        # Group by scene type
        result.groups = self._group_clips()

        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()

        return result

    def _analyze_clip(self, file_path: str) -> AnalyzedClip:
        """Analyze a single video clip"""
        filename = os.path.basename(file_path)

        clip = AnalyzedClip(
            original_path=file_path,
            original_name=filename
        )

        # Get technical metadata via FFprobe
        if self.ffprobe and self.ffprobe.is_available():
            metadata = self.ffprobe.analyze_video(file_path)
            if metadata:
                clip.duration = metadata.get("duration", 0)
                clip.resolution = (metadata.get("width", 1920), metadata.get("height", 1080))
                clip.frame_rate = metadata.get("frame_rate", 24.0)

        # Classify content based on filename patterns
        clip.scene_type, clip.shot_type = self._classify_from_filename(filename)

        # If filename doesn't give us info, use file characteristics
        if clip.scene_type == "unknown":
            clip.scene_type = self._classify_from_metadata(clip)

        return clip

    def _classify_from_filename(self, filename: str) -> Tuple[str, str]:
        """Try to classify clip from filename patterns"""
        name_lower = filename.lower()

        scene_type = "unknown"
        shot_type = "unknown"

        # Drone detection
        if any(x in name_lower for x in ["drone", "aerial", "dji", "mavic"]):
            scene_type = "drone"
            if "orbit" in name_lower:
                scene_type = "drone_orbit"
            elif "fly" in name_lower or "over" in name_lower:
                scene_type = "drone_flyover"
            shot_type = "aerial"

        # Room detection from filename
        room_patterns = {
            "kitchen": "kitchen",
            "living": "living_room",
            "family": "family_room",
            "dining": "dining_room",
            "bedroom": "bedroom",
            "master": "primary_bedroom",
            "primary": "primary_bedroom",
            "bathroom": "bathroom",
            "bath": "bathroom",
            "entry": "entry",
            "foyer": "foyer",
            "office": "office",
            "laundry": "laundry",
            "garage": "garage",
            "backyard": "backyard",
            "pool": "pool",
            "patio": "patio",
            "exterior": "exterior",
            "front": "exterior_front",
        }

        for pattern, room in room_patterns.items():
            if pattern in name_lower:
                scene_type = room
                break

        # Shot type detection
        if any(x in name_lower for x in ["walk", "through", "tour"]):
            shot_type = "walk-through"
        elif any(x in name_lower for x in ["static", "tripod", "locked"]):
            shot_type = "static"
        elif any(x in name_lower for x in ["detail", "insert", "close"]):
            shot_type = "detail"
        elif any(x in name_lower for x in ["gimbal", "steady"]):
            shot_type = "gimbal"

        return scene_type, shot_type

    def _classify_from_metadata(self, clip: AnalyzedClip) -> str:
        """Classify based on technical characteristics"""
        # Very short clips might be inserts/details
        if clip.duration < 5:
            return "detail"

        # Very long clips might be walk-throughs
        if clip.duration > 60:
            return "walk-through"

        # 4K+ resolution with specific aspect ratios might indicate drone
        if clip.resolution[0] >= 3840:
            # Could be drone, but not definitive
            pass

        return "unknown"

    def _sort_clips(self):
        """Sort clips by scene type order"""
        def sort_key(clip):
            order = self.SCENE_ORDER.get(clip.scene_type, 999)
            return (order, clip.original_name)

        self.clips.sort(key=sort_key)

        # Assign sort order
        for idx, clip in enumerate(self.clips):
            clip.sort_order = idx + 1

    def _assign_names(self):
        """Assign new standardized names to clips"""
        scene_counters = {}

        for clip in self.clips:
            scene = clip.scene_type

            # Get counter for this scene type
            if scene not in scene_counters:
                scene_counters[scene] = 0
            scene_counters[scene] += 1
            counter = scene_counters[scene]

            # Build new name
            scene_display = scene.replace("_", " ").title().replace(" ", "_")
            ext = os.path.splitext(clip.original_name)[1]

            # Format: 01_Drone_Establishing.MOV
            clip.new_name = f"{clip.sort_order:02d}_{scene_display}{ext}"

    def _group_clips(self) -> Dict[str, List[str]]:
        """Group clips by scene type"""
        groups = {}

        for clip in self.clips:
            scene = clip.scene_type
            if scene not in groups:
                groups[scene] = []
            groups[scene].append(clip.original_name)

        return groups

    def get_sorted_clips(self) -> List[AnalyzedClip]:
        """Get clips in sorted order"""
        return self.clips

    def export_sorted_files(self, output_folder: str, copy_files: bool = True) -> Dict[str, str]:
        """
        Export sorted/renamed files to output folder.
        Returns mapping of original -> new paths.
        """
        os.makedirs(output_folder, exist_ok=True)

        mapping = {}

        for clip in self.clips:
            new_path = os.path.join(output_folder, clip.new_name)

            if copy_files:
                shutil.copy2(clip.original_path, new_path)

            mapping[clip.original_path] = new_path

        return mapping


# ============================================================================
# PHOTO SORTER
# ============================================================================

class PhotoSorter:
    """Analyzes and sorts photos"""

    # Photo sort order for real estate
    PHOTO_ORDER = {
        "drone": 1,
        "aerial": 2,
        "exterior_front": 10,
        "exterior": 11,
        "entry": 20,
        "living_room": 30,
        "family_room": 31,
        "dining_room": 40,
        "kitchen": 50,
        "primary_bedroom": 60,
        "primary_bathroom": 61,
        "bedroom": 70,
        "bathroom": 71,
        "office": 80,
        "laundry": 81,
        "garage": 85,
        "backyard": 90,
        "pool": 91,
        "patio": 92,
        "twilight": 95,
        "lifestyle": 96,
        "other": 100,
        "unknown": 999,
    }

    def __init__(self):
        self.photos: List[AnalyzedPhoto] = []

    def analyze_folder(self, folder_path: str, photo_type: str = "interior",
                       progress_callback=None) -> FolderAnalysisResult:
        """
        Analyze all photos in a folder.
        photo_type: "interior", "drone", "twilight", "lifestyle"
        """
        result = FolderAnalysisResult(
            folder_path=folder_path,
            folder_type=f"raw_{photo_type}"
        )

        start_time = datetime.now()

        # Get photo files
        photo_files = []
        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    ext = os.path.splitext(item)[1].lower()
                    if ext in PHOTO_EXTENSIONS:
                        photo_files.append(item_path)
        except Exception as e:
            result.errors.append(f"Error scanning folder: {str(e)}")
            return result

        result.total_files = len(photo_files)
        photo_files.sort()

        # Analyze each photo
        for idx, file_path in enumerate(photo_files):
            filename = os.path.basename(file_path)

            if progress_callback:
                progress_callback(idx + 1, len(photo_files), filename)

            photo = AnalyzedPhoto(
                original_path=file_path,
                original_name=filename,
                photo_type=photo_type
            )

            # Basic classification from filename
            photo.room_type = self._classify_from_filename(filename)

            self.photos.append(photo)
            result.photos.append(photo)
            result.analyzed_files += 1

        # Sort photos
        self._sort_photos(photo_type)

        # Assign new names
        self._assign_names(photo_type)

        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()

        return result

    def _classify_from_filename(self, filename: str) -> str:
        """Classify photo from filename"""
        name_lower = filename.lower()

        room_patterns = {
            "kitchen": "kitchen",
            "living": "living_room",
            "family": "family_room",
            "dining": "dining_room",
            "bedroom": "bedroom",
            "master": "primary_bedroom",
            "primary": "primary_bedroom",
            "bathroom": "bathroom",
            "bath": "bathroom",
            "entry": "entry",
            "foyer": "entry",
            "office": "office",
            "laundry": "laundry",
            "garage": "garage",
            "backyard": "backyard",
            "pool": "pool",
            "patio": "patio",
            "exterior": "exterior",
            "front": "exterior_front",
        }

        for pattern, room in room_patterns.items():
            if pattern in name_lower:
                return room

        return "unknown"

    def _sort_photos(self, photo_type: str):
        """Sort photos by room type order"""
        def sort_key(photo):
            order = self.PHOTO_ORDER.get(photo.room_type, 999)
            return (order, photo.original_name)

        self.photos.sort(key=sort_key)

        for idx, photo in enumerate(self.photos):
            photo.sort_order = idx + 1

    def _assign_names(self, photo_type: str):
        """Assign standardized names"""
        prefix_map = {
            "interior": "Photo",
            "drone": "Drone",
            "twilight": "Twilight",
            "lifestyle": "Lifestyle",
        }
        prefix = prefix_map.get(photo_type, "Photo")

        for idx, photo in enumerate(self.photos):
            ext = os.path.splitext(photo.original_name)[1]
            photo.new_name = f"{prefix}-{idx + 1}{ext}"

    def export_sorted_files(self, output_folder: str, copy_files: bool = True) -> Dict[str, str]:
        """Export sorted/renamed files"""
        os.makedirs(output_folder, exist_ok=True)

        mapping = {}

        for photo in self.photos:
            new_path = os.path.join(output_folder, photo.new_name)

            if copy_files:
                shutil.copy2(photo.original_path, new_path)

            mapping[photo.original_path] = new_path

        return mapping


# ============================================================================
# LOCAL FOLDER PROCESSOR - Main Interface
# ============================================================================

class LocalFolderProcessor:
    """
    Main interface for processing local folders.
    Combines scanning, analysis, sorting, and export.
    """

    def __init__(self):
        self.scanner = FolderScanner()
        self.video_sorter = None
        self.photo_sorter = None
        self.last_scan = None

    def scan_folder(self, folder_path: str) -> Dict[str, Any]:
        """Scan a folder and return detected content"""
        self.last_scan = self.scanner.scan_path(folder_path)
        return self.last_scan

    def process_videos(self, folder_path: str, progress_callback=None) -> FolderAnalysisResult:
        """Process video folder - analyze, sort, prepare for export"""
        self.video_sorter = VideoSorter()
        return self.video_sorter.analyze_folder(folder_path, progress_callback)

    def process_photos(self, folder_path: str, photo_type: str = "interior",
                       progress_callback=None) -> FolderAnalysisResult:
        """Process photo folder - analyze, sort, prepare for export"""
        self.photo_sorter = PhotoSorter()
        return self.photo_sorter.analyze_folder(folder_path, photo_type, progress_callback)

    def export_sorted_videos(self, output_folder: str, copy_files: bool = True) -> Dict[str, str]:
        """Export sorted video files"""
        if not self.video_sorter:
            return {}
        return self.video_sorter.export_sorted_files(output_folder, copy_files)

    def export_sorted_photos(self, output_folder: str, copy_files: bool = True) -> Dict[str, str]:
        """Export sorted photo files"""
        if not self.photo_sorter:
            return {}
        return self.photo_sorter.export_sorted_files(output_folder, copy_files)

    def generate_video_xml(self, output_folder: str, project_name: str = "Timeline X Export") -> Dict[str, str]:
        """Generate XML timelines for video clips"""
        if not self.video_sorter or not TIMELINE_X_AVAILABLE:
            return {}

        os.makedirs(output_folder, exist_ok=True)

        # Create Timeline X instance
        timeline_x = TimelineX()
        timeline_x.set_format(ContentFormat.REAL_ESTATE)

        # Add clips
        for clip in self.video_sorter.clips:
            timeline_clip = Clip(
                id=f"clip_{clip.sort_order}",
                file_path=clip.original_path,
                filename=clip.new_name,
                duration=clip.duration,
                in_point=0.0,
                out_point=clip.duration,
                resolution=clip.resolution,
                frame_rate=clip.frame_rate,
            )
            timeline_x.add_clip(timeline_clip)

        # Generate timeline
        timeline_x.generate_timeline()

        # Export XMLs
        exports = {}

        try:
            davinci_path = os.path.join(output_folder, f"{project_name}_davinci.xml")
            timeline_x.export_davinci(davinci_path)
            exports["davinci"] = davinci_path
        except Exception as e:
            pass

        try:
            fcpxml_path = os.path.join(output_folder, f"{project_name}.fcpxml")
            timeline_x.export_fcpxml(fcpxml_path)
            exports["fcpxml"] = fcpxml_path
        except Exception as e:
            pass

        try:
            premiere_path = os.path.join(output_folder, f"{project_name}_premiere.xml")
            timeline_x.export_premiere(premiere_path)
            exports["premiere"] = premiere_path
        except Exception as e:
            pass

        return exports


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


# ============================================================================
# MAIN - Testing
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Local Folder Processor")
    print("  By Aerial Canvas")
    print("=" * 60 + "\n")

    # Check dependencies
    print("Dependencies:")
    print(f"  FFprobe Analyzer: {'Available' if ANALYZER_AVAILABLE else 'NOT FOUND'}")
    print(f"  Timeline X:       {'Available' if TIMELINE_X_AVAILABLE else 'NOT FOUND'}")

    print("\nLocal Folder Processor ready.")
    print("Use LocalFolderProcessor class to scan and process folders.")
