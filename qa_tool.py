"""
Proof by Aerial Canvas - Beta
Automated quality assurance for video and photo deliverables
"""

import streamlit as st
import subprocess
import json
import re
import os
import tempfile
import requests
import sqlite3
import time
import hashlib
import zipfile
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
import numpy as np

# User authentication and tracking
from database import user_db, learning_db

# Google OAuth (optional - for team authentication)
try:
    from streamlit_oauth import OAuth2Component
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

# Dropbox SDK (optional - for shared link integration)
try:
    import dropbox
    from dropbox.exceptions import AuthError, ApiError
    from dropbox.files import FolderMetadata, FileMetadata
    from dropbox.oauth import DropboxOAuth2FlowNoRedirect
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False

# Dropbox API credentials (Aerial Canvas QA Tool)
DROPBOX_APP_KEY = "ta9b2km6af2j2h4"
DROPBOX_APP_SECRET = "wknntqselzkuqof"
# Permanent refresh token for Aerial Canvas team account (hello@aerialcanvas.com)
DROPBOX_REFRESH_TOKEN = "RuUb2zgbOpcAAAAAAAAAASo9Ers5N4blrjJQwmxT1dR946ljVQv-HrNvC6c8D8-Q"

def get_dropbox_auth_flow():
    """Create Dropbox OAuth flow"""
    if not DROPBOX_AVAILABLE:
        return None
    return DropboxOAuth2FlowNoRedirect(
        DROPBOX_APP_KEY,
        DROPBOX_APP_SECRET,
        token_access_type='offline'
    )

def get_dropbox_client():
    """Get authenticated Dropbox client using the team refresh token (no user auth needed)"""
    if not DROPBOX_AVAILABLE:
        return None
    try:
        # Use refresh token - automatically gets fresh access tokens
        dbx = dropbox.Dropbox(
            oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
            app_key=DROPBOX_APP_KEY,
            app_secret=DROPBOX_APP_SECRET
        )
        # Verify connection works
        dbx.users_get_current_account()
        return dbx
    except Exception as e:
        print(f"Dropbox connection error: {e}")
        return None

def extract_dropbox_path_from_link(shared_link):
    """Extract the folder path from a Dropbox shared link"""
    # Shared links don't directly give us the path, we need to use the API
    # to get the shared link metadata
    return shared_link

def organize_in_dropbox(dbx, source_folder_path, project_name, clips_data, export_format):
    """
    Create organized folder structure in Dropbox with renamed files.

    Args:
        dbx: Authenticated Dropbox client
        source_folder_path: Path to source folder in Dropbox
        project_name: Name for the organized project folder
        clips_data: List of analyzed clip data with room classifications
        export_format: Export format for XML generation

    Returns:
        Path to the organized folder in Dropbox
    """
    # Create destination folder path (sibling to source)
    parent_path = os.path.dirname(source_folder_path.rstrip('/'))
    dest_folder = f"{parent_path}/{project_name}_Organized"

    # Create the destination folder
    try:
        dbx.files_create_folder_v2(dest_folder)
    except ApiError as e:
        if 'path/conflict/folder' in str(e):
            # Folder exists, add timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_folder = f"{parent_path}/{project_name}_Organized_{timestamp}"
            dbx.files_create_folder_v2(dest_folder)
        else:
            raise e

    # Sort clips by room type
    room_order = list(ROOM_TYPES.keys())
    sorted_clips = sorted(clips_data,
        key=lambda c: (room_order.index(c['room_type']) if c['room_type'] in room_order else 99, c['filename']))

    # Copy and rename files
    exported_clips = []
    room_counters = {}

    for idx, clip in enumerate(sorted_clips):
        room = clip['room_type']
        room_name = ROOM_TYPES.get(room, {}).get('name', room).replace('/', '-').replace(' ', '')

        # Track counter per room
        if room not in room_counters:
            room_counters[room] = 1
        else:
            room_counters[room] += 1

        # Generate new filename
        ext = os.path.splitext(clip['filename'])[1]
        new_filename = f"{idx+1:02d}_{room_name}_{room_counters[room]:02d}{ext}"

        # Copy file in Dropbox
        source_path = clip['dropbox_path']
        dest_path = f"{dest_folder}/{new_filename}"

        try:
            dbx.files_copy_v2(source_path, dest_path)
            exported_clip = clip.copy()
            exported_clip['path'] = new_filename
            exported_clip['filename'] = new_filename
            exported_clips.append(exported_clip)
        except ApiError as e:
            # Skip files that can't be copied
            pass

    # Generate and upload XML
    if export_format == "Final Cut Pro X":
        xml_content = generate_fcpxml(exported_clips, project_name)
        xml_filename = f"{project_name}.fcpxml"
    else:
        xml_content = generate_premiere_xml(exported_clips, project_name)
        xml_filename = f"{project_name}.xml"

    # Upload XML to Dropbox
    xml_path = f"{dest_folder}/{xml_filename}"
    dbx.files_upload(xml_content.encode('utf-8'), xml_path)

    return dest_folder, len(exported_clips), xml_filename


def organize_photos_in_dropbox(dbx, source_shared_link, sorted_photos, progress_callback=None):
    """
    Create organized folder in Dropbox with renamed photos.

    Args:
        dbx: Authenticated Dropbox client
        source_shared_link: Original Dropbox shared link
        sorted_photos: List of sorted photo data with new filenames
        progress_callback: Optional function(current, total, filename) for progress updates

    Returns:
        (dest_folder_path, num_photos_copied) tuple
    """
    try:
        # Get the source folder path from shared link
        if progress_callback:
            progress_callback(0, len(sorted_photos), "Connecting to Dropbox...")

        link_meta = dbx.sharing_get_shared_link_metadata(source_shared_link)
        source_path = link_meta.path_lower

        # Create destination folder (sibling to source with _Sorted suffix)
        parent_path = os.path.dirname(source_path.rstrip('/'))
        folder_name = os.path.basename(source_path.rstrip('/'))
        dest_folder = f"{parent_path}/{folder_name}_Sorted"

        if progress_callback:
            progress_callback(0, len(sorted_photos), "Creating destination folder...")

        # Create the destination folder
        try:
            dbx.files_create_folder_v2(dest_folder)
        except ApiError as e:
            if 'path/conflict/folder' in str(e):
                # Folder exists, add timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_folder = f"{parent_path}/{folder_name}_Sorted_{timestamp}"
                dbx.files_create_folder_v2(dest_folder)
            else:
                raise e

        # Copy and rename each photo
        copied_count = 0
        total_photos = len(sorted_photos)
        for i, photo in enumerate(sorted_photos):
            try:
                # Source path in Dropbox
                original_filename = photo['filename']
                source_file_path = f"{source_path}/{original_filename}"

                # New filename
                _, ext = os.path.splitext(original_filename)
                new_filename = f"{photo['new_filename']}{ext}"
                dest_file_path = f"{dest_folder}/{new_filename}"

                if progress_callback:
                    progress_callback(i + 1, total_photos, f"Copying {new_filename}...")

                # Copy file
                dbx.files_copy_v2(source_file_path, dest_file_path)
                copied_count += 1
            except ApiError as e:
                # Skip files that can't be copied
                print(f"Could not copy {original_filename}: {e}")
                continue

        return dest_folder, copied_count

    except Exception as e:
        raise Exception(f"Dropbox error: {str(e)}")


# YOLO Object Detection (for intelligent room classification)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# YOLO model cache
_yolo_model = None

def get_yolo_model():
    """Load YOLO model (cached for performance)"""
    global _yolo_model
    if _yolo_model is None and YOLO_AVAILABLE:
        try:
            # Use YOLOv8 nano - smallest/fastest, perfect for web deployment
            _yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Could not load YOLO model: {e}")
            return None
    return _yolo_model


# Object-to-Room mapping: which objects indicate which rooms
OBJECT_ROOM_MAPPING = {
    # Bathroom indicators
    'toilet': {'bathroom': 0.9},
    'sink': {'bathroom': 0.4, 'kitchen': 0.3},  # sinks in both
    'bathtub': {'bathroom': 0.9},
    'toothbrush': {'bathroom': 0.7},

    # Bedroom indicators
    'bed': {'bedroom': 0.9, 'primary_bedroom': 0.5},

    # Living room indicators (enhanced based on training images)
    # Primary indicators - high confidence
    'couch': {'living_room': 0.9},
    'sofa': {'living_room': 0.9},
    'tv': {'living_room': 0.7, 'bedroom': 0.3},
    'remote': {'living_room': 0.5},
    # Secondary indicators - coffee tables, seating arrangements
    'vase': {'living_room': 0.5, 'dining_room': 0.3},
    'potted plant': {'living_room': 0.4, 'exterior_front': 0.3, 'backyard': 0.3},
    'book': {'living_room': 0.4, 'office': 0.4},
    'clock': {'living_room': 0.3, 'office': 0.2},
    # Fireplace-related (common living room focal point)
    'fire': {'living_room': 0.6},
    # Art and decor
    'picture frame': {'living_room': 0.4, 'bedroom': 0.3},
    'painting': {'living_room': 0.4},
    # Lighting
    'lamp': {'living_room': 0.4, 'bedroom': 0.4},

    # Kitchen indicators
    'oven': {'kitchen': 0.9},
    'microwave': {'kitchen': 0.8},
    'refrigerator': {'kitchen': 0.9},
    'toaster': {'kitchen': 0.7},
    'knife': {'kitchen': 0.5},
    'spoon': {'kitchen': 0.4},
    'fork': {'kitchen': 0.4},
    'bowl': {'kitchen': 0.3, 'dining_room': 0.3},
    'cup': {'kitchen': 0.3},
    'bottle': {'kitchen': 0.2},

    # Dining room indicators
    'dining table': {'dining_room': 0.8, 'kitchen': 0.3},
    'wine glass': {'dining_room': 0.5},
    'chair': {'living_room': 0.4, 'dining_room': 0.3, 'office': 0.3},

    # Office indicators
    'laptop': {'office': 0.6, 'bedroom': 0.2},
    'keyboard': {'office': 0.7},
    'mouse': {'office': 0.6},

    # Garage indicators
    'car': {'garage': 0.7, 'exterior_front': 0.4},
    'truck': {'garage': 0.6, 'exterior_front': 0.4},
    'bicycle': {'garage': 0.5},
    'motorcycle': {'garage': 0.6},

    # Exterior indicators
    'bench': {'backyard': 0.5, 'exterior_front': 0.3, 'exterior_rear': 0.4},
    'bird': {'exterior_front': 0.4, 'backyard': 0.4, 'exterior_rear': 0.3},
    'dog': {'backyard': 0.4, 'exterior_rear': 0.3, 'living_room': 0.3},
    'cat': {'living_room': 0.4, 'backyard': 0.2},

    # Backyard/exterior rear indicators
    'umbrella': {'backyard': 0.5, 'pool': 0.5, 'exterior_rear': 0.3},
    'surfboard': {'pool': 0.4, 'garage': 0.3},
    'sports ball': {'backyard': 0.4, 'yard': 0.3},
    'frisbee': {'backyard': 0.5, 'yard': 0.4},
    'kite': {'backyard': 0.4, 'exterior_rear': 0.3},
    'tennis racket': {'backyard': 0.4},
    'skateboard': {'backyard': 0.3, 'garage': 0.4},

    # Pool indicators
    'person': {'pool': 0.1, 'living_room': 0.1},  # Low weight, common everywhere
}


def detect_objects_in_frame(frame) -> Dict[str, float]:
    """
    Use YOLO to detect objects in a frame.
    Returns dict of {object_name: confidence}
    """
    model = get_yolo_model()
    if model is None:
        return {}

    try:
        # Run detection (verbose=False to suppress output)
        results = model(frame, verbose=False)

        detected = {}
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                # Keep highest confidence for each object type
                if class_name not in detected or confidence > detected[class_name]:
                    detected[class_name] = confidence

        return detected
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return {}


def score_room_from_objects(detected_objects: Dict[str, float]) -> Dict[str, float]:
    """
    Convert detected objects into room type scores.
    """
    room_scores = {}

    for obj_name, obj_confidence in detected_objects.items():
        obj_lower = obj_name.lower()

        if obj_lower in OBJECT_ROOM_MAPPING:
            room_weights = OBJECT_ROOM_MAPPING[obj_lower]
            for room, weight in room_weights.items():
                score = obj_confidence * weight
                room_scores[room] = room_scores.get(room, 0) + score

    return room_scores


# ============================================================================
# DROPBOX INTEGRATION
# ============================================================================

def get_dropbox_client():
    """
    Get Dropbox client using access token from environment or session state.

    To set up:
    1. Go to https://www.dropbox.com/developers/apps
    2. Create an app with "Full Dropbox" access
    3. Generate an access token
    4. Set DROPBOX_ACCESS_TOKEN environment variable or enter in the app
    """
    if not DROPBOX_AVAILABLE:
        return None

    # Check session state first (user-entered token)
    token = st.session_state.get('dropbox_token', '')

    # Fall back to environment variable
    if not token:
        token = os.environ.get('DROPBOX_ACCESS_TOKEN', '')

    if not token:
        return None

    try:
        dbx = dropbox.Dropbox(token)
        # Verify token works
        dbx.users_get_current_account()
        return dbx
    except Exception:
        return None


def parse_dropbox_shared_link(url: str) -> dict:
    """
    Parse a Dropbox shared link to extract useful info.
    Handles various Dropbox URL formats.
    """
    result = {
        "url": url,
        "is_valid": False,
        "is_folder": False,
        "direct_url": None
    }

    if not url:
        return result

    # Check if it's a Dropbox URL
    if "dropbox.com" not in url.lower():
        return result

    result["is_valid"] = True

    # Detect folder vs file
    if "/scl/fo/" in url or "/sh/" in url:
        result["is_folder"] = True
    elif "/scl/fi/" in url or "/s/" in url:
        result["is_folder"] = False

    # Create direct download URL (for files)
    if "?dl=0" in url:
        result["direct_url"] = url.replace("?dl=0", "?dl=1")
    elif "?dl=1" in url:
        result["direct_url"] = url
    elif "?" in url:
        result["direct_url"] = url + "&dl=1"
    else:
        result["direct_url"] = url + "?dl=1"

    return result


def list_dropbox_shared_folder(shared_link: str, dbx_client=None) -> List[Dict]:
    """
    List files in a shared Dropbox folder.
    Returns list of file info dicts with name, path, size, etc.
    """
    if not DROPBOX_AVAILABLE:
        return []

    if not dbx_client:
        dbx_client = get_dropbox_client()

    if not dbx_client:
        return []

    files = []
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.m4v'}

    try:
        # Create shared link metadata
        shared_link_metadata = dropbox.files.SharedLink(url=shared_link)

        # List folder contents
        result = dbx_client.files_list_folder(
            path="",
            shared_link=shared_link_metadata
        )

        while True:
            for entry in result.entries:
                if isinstance(entry, FileMetadata):
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in video_extensions:
                        files.append({
                            "name": entry.name,
                            "path": entry.path_display,
                            "size": entry.size,
                            "id": entry.id,
                            "content_hash": entry.content_hash,
                            "is_video": True
                        })
                elif isinstance(entry, FolderMetadata):
                    # Could recursively list subfolders if needed
                    pass

            if not result.has_more:
                break

            result = dbx_client.files_list_folder_continue(result.cursor)

    except ApiError as e:
        st.error(f"Dropbox API error: {e}")
        return []
    except Exception as e:
        st.error(f"Error listing Dropbox folder: {e}")
        return []

    return files


def download_dropbox_file(shared_link: str, filename: str, dest_path: str, dbx_client=None) -> bool:
    """
    Download a specific file from a shared Dropbox folder.
    """
    if not DROPBOX_AVAILABLE:
        return False

    if not dbx_client:
        dbx_client = get_dropbox_client()

    if not dbx_client:
        return False

    try:
        shared_link_metadata = dropbox.files.SharedLink(url=shared_link)

        # Download file
        dbx_client.sharing_get_shared_link_file_to_file(
            download_path=dest_path,
            url=shared_link,
            path=f"/{filename}"
        )
        return True

    except Exception as e:
        # Try alternative method - direct download via URL modification
        try:
            # Construct direct download URL
            if "?" in shared_link:
                direct_url = shared_link.split("?")[0] + "?dl=1"
            else:
                direct_url = shared_link + "?dl=1"

            # For folder links, we need to construct file URL differently
            # This is a fallback that may not work for all cases
            response = requests.get(direct_url, stream=True, timeout=300)
            if response.status_code == 200:
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception:
            pass

        return False


def get_dropbox_download_link(shared_link: str, file_path: str, dbx_client=None) -> Optional[str]:
    """
    Get a temporary direct download link for a file in a shared folder.
    """
    if not DROPBOX_AVAILABLE or not dbx_client:
        return None

    try:
        # Get temporary link
        link_metadata = dbx_client.sharing_get_shared_link_file(
            url=shared_link,
            path=file_path
        )
        return link_metadata.link
    except Exception:
        return None

# ============================================================================
# AERIAL CANVAS BRAND COLORS (from webflow-SQUAD.txt)
# ============================================================================

# Primary brand colors
BRAND_ACCENT = "#7B8CDE"       # Primary accent (purple/periwinkle)
BRAND_ACCENT_HOVER = "#9BA8E8" # Accent hover state
BRAND_ACCENT_GLOW = "rgba(123,140,222,0.15)"  # Accent with transparency

# Background colors
BRAND_BG = "#000"              # Primary background (black)
BRAND_BG2 = "#0a0a0a"          # Secondary background
BRAND_CARD = "#111"            # Card background
BRAND_CARD2 = "#161616"        # Card secondary

# Text colors
BRAND_TEXT = "#fff"            # Primary text (white)
BRAND_TEXT2 = "#a1a1aa"        # Secondary text (muted)
BRAND_TEXT3 = "#71717a"        # Tertiary text (more muted)

# Border
BRAND_BORDER = "#1d1d1f"       # Border color

# Status colors (for pass/fail/warning - kept for clarity)
BRAND_GREEN = "#4ade80"        # Pass/success
BRAND_RED = "#ef4444"          # Fail/error
BRAND_YELLOW = "#f59e0b"       # Warning

# Aliases for backward compatibility
BRAND_PURPLE = BRAND_ACCENT
BRAND_GRAY = BRAND_TEXT2

# Custom SVG icons in brand colors
ICONS = {
    # Status icons
    'pass': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_GREEN}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M8 12l2.5 2.5L16 9"></path>
    </svg>''',

    'fail': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_RED}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M15 9l-6 6M9 9l6 6"></path>
    </svg>''',

    'warning': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_YELLOW}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2L2 22h20L12 2z"></path>
        <path d="M12 9v4M12 17h.01"></path>
    </svg>''',

    'info': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M12 16v-4M12 8h.01"></path>
    </svg>''',

    # Checkmark for passed items
    'check': f'''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{BRAND_GREEN}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5 12l5 5L20 7"></path>
    </svg>''',

    'x': f'''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{BRAND_RED}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
        <path d="M18 6L6 18M6 6l12 12"></path>
    </svg>''',

    # Feedback icons
    'thumbs_up': f'''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{BRAND_GREEN}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
    </svg>''',

    'thumbs_down': f'''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{BRAND_RED}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
    </svg>''',

    'note': f'''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="16" y1="13" x2="8" y2="13"></line>
        <line x1="16" y1="17" x2="8" y2="17"></line>
    </svg>''',

    # Mode icons
    'qa_tool': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <path d="M21 21l-4.35-4.35"></path>
        <path d="M8 11h6M11 8v6"></path>
    </svg>''',

    'calibration': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="18" y1="20" x2="18" y2="10"></line>
        <line x1="12" y1="20" x2="12" y2="4"></line>
        <line x1="6" y1="20" x2="6" y2="14"></line>
        <circle cx="18" cy="8" r="2"></circle>
        <circle cx="12" cy="4" r="2"></circle>
        <circle cx="6" cy="12" r="2"></circle>
    </svg>''',

    'folder': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
    </svg>''',

    # Action icons
    'lightbulb': f'''<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{BRAND_YELLOW}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.9V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.1A7 7 0 0 0 12 2z"></path>
    </svg>''',

    'file': f'''<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
    </svg>''',

    'clock': f'''<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <polyline points="12 6 12 12 16 14"></polyline>
    </svg>''',

    'refresh': f'''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="23 4 23 10 17 10"></polyline>
        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
    </svg>''',

    'trash': f'''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{BRAND_RED}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
    </svg>''',

    # Navigation
    'arrow_left': f'''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="19" y1="12" x2="5" y2="12"></line>
        <polyline points="12 19 5 12 12 5"></polyline>
    </svg>''',

    'arrow_right': f'''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="5" y1="12" x2="19" y2="12"></line>
        <polyline points="12 5 19 12 12 19"></polyline>
    </svg>''',

    # =============================================
    # ROOM TYPE ICONS - Brand styled (outline)
    # =============================================

    # Exterior Front - House outline
    'room_exterior': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
        <polyline points="9 22 9 12 15 12 15 22"></polyline>
    </svg>''',

    # Exterior Rear - Back of house
    'room_exterior_rear': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
        <path d="M9 22v-6h6v6"></path>
        <path d="M15 16h3M15 19h3"></path>
    </svg>''',

    # Backyard - Fence/grass
    'room_backyard': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 22v-8l2-2 2 2v8"></path>
        <path d="M10 22v-8l2-2 2 2v8"></path>
        <path d="M16 22v-8l2-2 2 2v8"></path>
        <path d="M2 14h20"></path>
        <path d="M7 22c0-2 1-3 2-3s2 1 2 3"></path>
    </svg>''',

    # Entry/Foyer - Door
    'room_entry': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M18 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z"></path>
        <circle cx="15" cy="12" r="1"></circle>
        <path d="M4 21V3"></path>
    </svg>''',

    # Living Room - Sofa/Couch
    'room_living': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v3"></path>
        <path d="M2 11v5a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-5a2 2 0 0 0-4 0v2H6v-2a2 2 0 0 0-4 0z"></path>
        <path d="M4 18v2M20 18v2"></path>
    </svg>''',

    # Kitchen - Chef hat / cooking
    'room_kitchen': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M6 13.87A4 4 0 0 1 7.41 6a5 5 0 0 1 9.18 0A4 4 0 0 1 18 13.87V21H6v-7.13z"></path>
        <line x1="6" y1="17" x2="18" y2="17"></line>
    </svg>''',

    # Dining Room - Table/utensils
    'room_dining': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"></path>
        <path d="M7 2v20"></path>
        <path d="M21 15V2v0a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3zm0 0v7"></path>
    </svg>''',

    # Primary Bedroom - Bed with crown/star
    'room_primary_bed': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M2 16V9a4 4 0 0 1 4-4h12a4 4 0 0 1 4 4v7"></path>
        <path d="M2 16h20v4H2z"></path>
        <path d="M6 12h4v4H6zM14 12h4v4h-4z"></path>
        <path d="M12 2l1 2h-2l1-2z"></path>
    </svg>''',

    # Bedroom - Bed
    'room_bedroom': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M2 16V9a4 4 0 0 1 4-4h12a4 4 0 0 1 4 4v7"></path>
        <path d="M2 16h20v4H2z"></path>
        <path d="M6 12h4v4H6zM14 12h4v4h-4z"></path>
    </svg>''',

    # Bathroom - Bathtub
    'room_bathroom': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 12h16a1 1 0 0 1 1 1v3a4 4 0 0 1-4 4H7a4 4 0 0 1-4-4v-3a1 1 0 0 1 1-1z"></path>
        <path d="M6 12V5a2 2 0 0 1 2-2h1a2 2 0 0 1 2 2"></path>
        <circle cx="17" cy="7" r="2"></circle>
    </svg>''',

    # Office - Briefcase/desk
    'room_office': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
        <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>
    </svg>''',

    # Backyard/Pool - Water waves
    'room_pool': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M2 12c.6-.5 1.2-1 2.5-1 2.5 0 2.5 2 5 2 2.6 0 2.6-2 5-2 2.5 0 2.5 2 5 2 1.3 0 1.9-.5 2.5-1"></path>
        <path d="M2 17c.6-.5 1.2-1 2.5-1 2.5 0 2.5 2 5 2 2.6 0 2.6-2 5-2 2.5 0 2.5 2 5 2 1.3 0 1.9-.5 2.5-1"></path>
        <path d="M2 7c.6-.5 1.2-1 2.5-1 2.5 0 2.5 2 5 2 2.6 0 2.6-2 5-2 2.5 0 2.5 2 5 2 1.3 0 1.9-.5 2.5-1"></path>
    </svg>''',

    # Yard/Garden - Tree
    'room_yard': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 22v-7"></path>
        <path d="M9 22h6"></path>
        <path d="M12 15a7 7 0 0 0 7-7c0-2-1-4-3-5.5C14 1 12 1 12 1s-2 0-4 1.5C6 4 5 6 5 8a7 7 0 0 0 7 7z"></path>
    </svg>''',

    # Garage - Car
    'room_garage': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5 17a2 2 0 1 0 0-4 2 2 0 0 0 0 4zM19 17a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"></path>
        <path d="M5 15H3v-3.5a.5.5 0 0 1 .5-.5h1.7l1.3-3h10l1.3 3h1.7a.5.5 0 0 1 .5.5V15h-2"></path>
        <path d="M7 15h10"></path>
    </svg>''',

    # Laundry Room - Washer/dryer
    'room_laundry': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2"></rect>
        <circle cx="12" cy="13" r="5"></circle>
        <circle cx="12" cy="13" r="2"></circle>
        <circle cx="7" cy="6" r="1"></circle>
        <circle cx="10" cy="6" r="1"></circle>
    </svg>''',

    # ADU - Small house/cottage
    'room_adu': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 12l9-7 9 7"></path>
        <path d="M5 10v10h14V10"></path>
        <rect x="9" y="14" width="6" height="6"></rect>
        <path d="M12 14v6"></path>
        <path d="M9 17h6"></path>
    </svg>''',

    # Drone - Quadcopter
    'room_drone': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="5" cy="5" r="2"></circle>
        <circle cx="19" cy="5" r="2"></circle>
        <circle cx="5" cy="19" r="2"></circle>
        <circle cx="19" cy="19" r="2"></circle>
        <path d="M5 7v2M19 7v2M5 15v2M19 15v2"></path>
        <rect x="9" y="9" width="6" height="6" rx="1"></rect>
    </svg>''',

    # Detail - Sparkle/star
    'room_detail': f'''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2l2 6h6l-5 4 2 6-5-4-5 4 2-6-5-4h6l2-6z"></path>
    </svg>''',

    # Feature card icons
    'room_detection': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="7" height="7"></rect>
        <rect x="14" y="3" width="7" height="7"></rect>
        <rect x="14" y="14" width="7" height="7"></rect>
        <rect x="3" y="14" width="7" height="7"></rect>
    </svg>''',

    'best_moments': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
    </svg>''',

    'xml_export': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="16" y1="13" x2="8" y2="13"></line>
        <line x1="16" y1="17" x2="8" y2="17"></line>
    </svg>''',

    # Video Proof feature icons (stroke-width 2 for better visibility)
    'technical_scan': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="2" y="3" width="20" height="14" rx="2"></rect>
        <line x1="8" y1="21" x2="16" y2="21"></line>
        <line x1="12" y1="17" x2="12" y2="21"></line>
    </svg>''',

    'aspect_ratio': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="5" width="18" height="14" rx="2"></rect>
        <line x1="3" y1="9" x2="21" y2="9"></line>
        <line x1="9" y1="5" x2="9" y2="19"></line>
    </svg>''',

    'audio_analysis': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
        <line x1="12" y1="19" x2="12" y2="22"></line>
        <line x1="8" y1="22" x2="16" y2="22"></line>
    </svg>''',

    'color_grade': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <circle cx="12" cy="12" r="4"></circle>
        <circle cx="12" cy="12" r="1" fill="{BRAND_PURPLE}"></circle>
    </svg>''',

    'edit_quality': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="2" y="2" width="20" height="20" rx="2"></rect>
        <line x1="7" y1="2" x2="7" y2="22"></line>
        <line x1="17" y1="2" x2="17" y2="22"></line>
        <line x1="2" y1="12" x2="22" y2="12"></line>
    </svg>''',

    'compliance': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
    </svg>''',

    # Analysis mode icons
    'mode_timeline': f'''<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
    </svg>''',

    'mode_quick': f'''<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="23 4 23 10 17 10"></polyline>
        <polyline points="1 20 1 14 7 14"></polyline>
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
    </svg>''',

    'mode_full': f'''<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        <circle cx="11" cy="11" r="3"></circle>
    </svg>''',

    # Photo Proof feature icons (stroke-width 2 for better visibility)
    'sharpness_check': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        <line x1="11" y1="8" x2="11" y2="14"></line>
        <line x1="8" y1="11" x2="14" y2="11"></line>
    </svg>''',

    'exposure_color': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
    </svg>''',

    'composition': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2"></rect>
        <line x1="3" y1="9" x2="21" y2="9"></line>
        <line x1="3" y1="15" x2="21" y2="15"></line>
        <line x1="9" y1="3" x2="9" y2="21"></line>
        <line x1="15" y1="3" x2="15" y2="21"></line>
    </svg>''',

    'photo_icon': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2"></rect>
        <circle cx="8.5" cy="8.5" r="1.5"></circle>
        <path d="M21 15l-5-5L5 21"></path>
    </svg>''',

    'video_icon': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="2" y="4" width="14" height="16" rx="2"></rect>
        <path d="M16 8l4-2v12l-4-2"></path>
    </svg>''',

    'timeline': f'''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{BRAND_PURPLE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="2" y1="12" x2="22" y2="12"></line>
        <circle cx="6" cy="12" r="2" fill="{BRAND_PURPLE}"></circle>
        <circle cx="12" cy="12" r="2" fill="{BRAND_PURPLE}"></circle>
        <circle cx="18" cy="12" r="2" fill="{BRAND_PURPLE}"></circle>
    </svg>''',
}

def icon(name: str, size: int = None) -> str:
    """Get an SVG icon by name, optionally resized"""
    svg = ICONS.get(name, ICONS['info'])
    if size:
        svg = svg.replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"')
        svg = svg.replace('width="20"', f'width="{size}"').replace('height="20"', f'height="{size}"')
        svg = svg.replace('width="16"', f'width="{size}"').replace('height="16"', f'height="{size}"')
        svg = svg.replace('width="14"', f'width="{size}"').replace('height="14"', f'height="{size}"')
    return svg


def color_dot(color: str, size: int = 10) -> str:
    """Generate a colored dot HTML element for legends and indicators"""
    return f'<span style="display: inline-block; width: {size}px; height: {size}px; background: {color}; border-radius: 50%; margin-right: 4px;"></span>'


# ============================================================================
# TIME ESTIMATION
# ============================================================================

# Average time per step in seconds (will be calibrated over time)
STEP_TIMES = {
    'video': {
        'metadata': 1,
        'resolution': 0.5,
        'frame_rate': 0.5,
        'audio_presence': 0.5,
        'audio_levels': 3,
        'black_frames': 15,
        'log_footage': 20,
        'fade_out': 2,
        'stabilization': 25,
        'transitions': 10,
        'beat_sync': 15,
        'sound_design': 8,
        'filename': 0.5,
        'flow_rating': 0.5,
    },
    'photo': {
        'load': 1,
        'file_size': 0.5,
        'resolution': 0.5,
        'sharpness': 3,  # Now includes regional analysis
        'noise': 2,
        'hdr_blending': 2,  # HDR artifact detection
        'export_quality': 2,  # Compression artifacts, glitches
        'exposure': 1,
        'contrast': 1,
        'saturation': 1,
        'white_balance': 1,
        'ai_processing': 1,
        'reflections': 3,
        'staging': 2,  # Bathroom staging check
        'filename': 0.5,
    }
}

def get_total_estimated_time(file_type: str) -> int:
    """Get total estimated time in seconds for a file type"""
    return sum(STEP_TIMES.get(file_type, {}).values())

def format_time_remaining(seconds: float) -> str:
    """Format seconds into a readable time string"""
    if seconds < 60:
        return f"{int(seconds)}s"
    else:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"

class ProgressTracker:
    """Track progress with time estimation"""

    def __init__(self, file_type: str, progress_bar, status_container):
        self.file_type = file_type
        self.progress_bar = progress_bar
        self.status_container = status_container
        self.start_time = time.time()
        self.step_times = STEP_TIMES.get(file_type, {})
        self.total_estimated = get_total_estimated_time(file_type)
        self.completed_steps = []

    def update(self, step: int, total: int, message: str, step_name: str = None):
        """Update progress with time estimation"""
        elapsed = time.time() - self.start_time
        progress = step / total

        # Calculate estimated remaining time
        if step > 1 and elapsed > 0:
            # Use actual elapsed time to estimate remaining
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
        else:
            # Use pre-calculated estimates
            remaining_steps = list(self.step_times.keys())[step:]
            remaining = sum(self.step_times.get(s, 2) for s in remaining_steps)

        self.progress_bar.progress(progress)

        time_str = format_time_remaining(max(0, remaining))
        clock_icon = icon('clock', 14)
        self.status_container.markdown(f'''
            <div style="display: flex; align-items: center; gap: 8px;">
                {clock_icon}
                <span style="color: {BRAND_PURPLE}; font-size: 13px;">{message}</span>
                <span style="color: {BRAND_GRAY}; font-size: 12px; margin-left: auto;">~{time_str} remaining</span>
            </div>
        ''', unsafe_allow_html=True)

def render_progress_status(message: str, remaining_seconds: float = None) -> str:
    """Render a progress status message with optional time remaining"""
    clock_icon = icon('clock', 12)
    if remaining_seconds is not None and remaining_seconds > 0:
        time_str = format_time_remaining(remaining_seconds)
        return f'''<div style="display: flex; align-items: center; gap: 8px;">
            {clock_icon}
            <span style="color: {BRAND_PURPLE}; font-size: 13px;">{message}</span>
            <span style="color: {BRAND_GRAY}; font-size: 12px; margin-left: auto;">~{time_str} remaining</span>
        </div>'''
    else:
        return f'''<div style="display: flex; align-items: center; gap: 8px;">
            {clock_icon}
            <span style="color: {BRAND_PURPLE}; font-size: 13px;">{message}</span>
        </div>'''

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class QAIssue:
    """Represents a single QA issue found"""
    check_name: str
    status: str  # "pass", "fail", "warning", "info"
    message: str
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    expected: Optional[str] = None
    found: Optional[str] = None
    action: Optional[str] = None
    # Preview support - base64 encoded image showing the issue
    preview_image: Optional[str] = None  # Base64 encoded JPEG
    preview_coords: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) bounding box for photo issues

@dataclass
class QAReport:
    """Complete QA report for a file"""
    filename: str
    file_type: str  # "video" or "photo"
    overall_status: str  # "pass" or "fail"
    issues: List[QAIssue]
    metadata: dict

# ============================================================================
# PREVIEW IMAGE HELPERS
# ============================================================================

def capture_video_frame(video_path: str, timestamp: float) -> Optional[str]:
    """
    Capture a frame from video at specific timestamp and return as base64 JPEG.
    Returns None if capture fails.
    """
    import base64
    try:
        # Use ffmpeg to extract frame at timestamp
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            'ffmpeg', '-y', '-ss', str(timestamp),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '3',  # Good quality JPEG
            '-vf', 'scale=640:-1',  # Resize for preview
            tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)

        if result.returncode == 0 and os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                img_data = f.read()
            os.unlink(tmp_path)
            return base64.b64encode(img_data).decode('utf-8')

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None
    except Exception:
        return None


def create_highlighted_preview(image, coords: Tuple[int, int, int, int], color=(123, 140, 222)) -> Optional[str]:
    """
    Create a preview image with a highlighted bounding box around the issue.
    coords: (x, y, width, height)
    Returns base64 encoded JPEG.
    """
    import base64
    import cv2
    try:
        # Make a copy to draw on
        preview = image.copy()
        x, y, w, h = coords

        # Draw rectangle around the issue
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 3)

        # Add a semi-transparent overlay
        overlay = preview.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.2, preview, 0.8, 0, preview)

        # Resize for preview (max 800px wide)
        h_img, w_img = preview.shape[:2]
        if w_img > 800:
            scale = 800 / w_img
            preview = cv2.resize(preview, (800, int(h_img * scale)))

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return None


def create_photo_preview(image, scale: float = 0.5) -> Optional[str]:
    """
    Create a scaled preview of a photo.
    Returns base64 encoded JPEG.
    """
    import base64
    import cv2
    try:
        h, w = image.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w > 800:
            new_w = 800
            new_h = int(h * (800 / w))

        preview = cv2.resize(image, (new_w, new_h))
        _, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return None


# ============================================================================
# FEEDBACK & LEARNING DATABASE
# ============================================================================

class FeedbackDB:
    """SQLite database for storing feedback and learning from it"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Store in same directory as the app
            db_path = os.path.join(os.path.dirname(__file__), 'qa_feedback.db')
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Feedback table - stores individual ratings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                file_type TEXT,
                check_name TEXT,
                original_status TEXT,
                original_message TEXT,
                feedback_type TEXT,
                notes TEXT
            )
        ''')

        # Threshold adjustments table - stores learned thresholds
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS thresholds (
                check_name TEXT PRIMARY KEY,
                current_value REAL,
                suggested_value REAL,
                false_positives INTEGER DEFAULT 0,
                false_negatives INTEGER DEFAULT 0,
                last_updated TEXT
            )
        ''')

        # File ratings table - for batch rating mode
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                file_type TEXT,
                overall_rating TEXT,
                notes TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def add_feedback(self, filename: str, file_type: str, check_name: str,
                     original_status: str, original_message: str,
                     feedback_type: str, notes: str = ""):
        """
        Add feedback for a specific check result.
        feedback_type: 'correct', 'false_positive', 'false_negative'
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO feedback (timestamp, filename, file_type, check_name,
                                  original_status, original_message, feedback_type, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), filename, file_type, check_name,
              original_status, original_message, feedback_type, notes))

        # Update threshold stats
        if feedback_type == 'false_positive':
            cursor.execute('''
                INSERT INTO thresholds (check_name, false_positives, last_updated)
                VALUES (?, 1, ?)
                ON CONFLICT(check_name) DO UPDATE SET
                    false_positives = false_positives + 1,
                    last_updated = ?
            ''', (check_name, datetime.now().isoformat(), datetime.now().isoformat()))
        elif feedback_type == 'false_negative':
            cursor.execute('''
                INSERT INTO thresholds (check_name, false_negatives, last_updated)
                VALUES (?, 1, ?)
                ON CONFLICT(check_name) DO UPDATE SET
                    false_negatives = false_negatives + 1,
                    last_updated = ?
            ''', (check_name, datetime.now().isoformat(), datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def add_file_rating(self, filename: str, file_type: str, rating: str, notes: str = ""):
        """Add overall file rating (for batch mode)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO file_ratings (timestamp, filename, file_type, overall_rating, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), filename, file_type, rating, notes))

        conn.commit()
        conn.close()

    def get_feedback_stats(self) -> Dict:
        """Get statistics about collected feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total feedback count
        cursor.execute('SELECT COUNT(*) FROM feedback')
        total_feedback = cursor.fetchone()[0]

        # Feedback by check
        cursor.execute('''
            SELECT check_name, feedback_type, COUNT(*)
            FROM feedback
            GROUP BY check_name, feedback_type
        ''')
        by_check = {}
        for row in cursor.fetchall():
            check_name, feedback_type, count = row
            if check_name not in by_check:
                by_check[check_name] = {'correct': 0, 'false_positive': 0, 'false_negative': 0}
            by_check[check_name][feedback_type] = count

        # Threshold stats
        cursor.execute('SELECT * FROM thresholds')
        thresholds = {}
        for row in cursor.fetchall():
            thresholds[row[0]] = {
                'current_value': row[1],
                'suggested_value': row[2],
                'false_positives': row[3],
                'false_negatives': row[4],
                'last_updated': row[5]
            }

        # File ratings
        cursor.execute('SELECT overall_rating, COUNT(*) FROM file_ratings GROUP BY overall_rating')
        file_ratings = dict(cursor.fetchall())

        # Recent feedback
        cursor.execute('''
            SELECT timestamp, filename, check_name, feedback_type
            FROM feedback
            ORDER BY timestamp DESC
            LIMIT 20
        ''')
        recent = cursor.fetchall()

        conn.close()

        return {
            'total_feedback': total_feedback,
            'by_check': by_check,
            'thresholds': thresholds,
            'file_ratings': file_ratings,
            'recent': recent
        }

    def get_check_accuracy(self, check_name: str) -> Dict:
        """Get accuracy stats for a specific check"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT feedback_type, COUNT(*)
            FROM feedback
            WHERE check_name = ?
            GROUP BY feedback_type
        ''', (check_name,))

        stats = {'correct': 0, 'false_positive': 0, 'false_negative': 0}
        for row in cursor.fetchall():
            stats[row[0]] = row[1]

        total = sum(stats.values())
        accuracy = (stats['correct'] / total * 100) if total > 0 else 0

        conn.close()

        return {
            'total': total,
            'correct': stats['correct'],
            'false_positives': stats['false_positive'],
            'false_negatives': stats['false_negative'],
            'accuracy': accuracy
        }

    def clear_all_feedback(self):
        """Clear all feedback data (for reset)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM feedback')
        cursor.execute('DELETE FROM thresholds')
        cursor.execute('DELETE FROM file_ratings')
        conn.commit()
        conn.close()

# Global feedback database instance
feedback_db = FeedbackDB()


class StatsTracker:
    """Track usage statistics and value metrics over time"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'qa_stats.db')
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize stats database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_name TEXT UNIQUE,
                stat_value INTEGER DEFAULT 0,
                last_updated TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_type TEXT,
                files_analyzed INTEGER,
                issues_found INTEGER,
                duration_seconds REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS room_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                original_room TEXT,
                corrected_room TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trim_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                original_in REAL,
                original_out REAL,
                corrected_in REAL,
                corrected_out REAL,
                total_duration REAL
            )
        ''')

        # Photo room detections - used to cross-train video detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS photo_room_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                folder_path TEXT,
                detected_room TEXT,
                confidence REAL,
                objects_detected TEXT
            )
        ''')

        # Initialize default stats if they don't exist
        default_stats = [
            'total_photos_analyzed',
            'total_videos_analyzed',
            'total_clips_analyzed',
            'total_issues_found',
            'total_room_corrections',
            'total_trim_corrections',
            'total_sessions'
        ]
        for stat in default_stats:
            cursor.execute('''
                INSERT OR IGNORE INTO stats (stat_name, stat_value, last_updated)
                VALUES (?, 0, datetime('now'))
            ''', (stat,))

        conn.commit()
        conn.close()

    def increment_stat(self, stat_name: str, amount: int = 1):
        """Increment a stat counter"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO stats (stat_name, stat_value, last_updated)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(stat_name) DO UPDATE SET
                stat_value = stat_value + ?,
                last_updated = datetime('now')
        ''', (stat_name, amount, amount))
        conn.commit()
        conn.close()

    def get_stat(self, stat_name: str) -> int:
        """Get a single stat value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT stat_value FROM stats WHERE stat_name = ?', (stat_name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0

    def get_all_stats(self) -> Dict:
        """Get all stats as a dictionary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT stat_name, stat_value FROM stats')
        results = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in results}

    def log_session(self, session_type: str, files_analyzed: int, issues_found: int, duration_seconds: float):
        """Log a QA session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO session_log (timestamp, session_type, files_analyzed, issues_found, duration_seconds)
            VALUES (datetime('now'), ?, ?, ?, ?)
        ''', (session_type, files_analyzed, issues_found, duration_seconds))
        conn.commit()
        conn.close()
        self.increment_stat('total_sessions')

    def log_room_correction(self, filename: str, original_room: str, corrected_room: str):
        """Log a room classification correction for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO room_corrections (timestamp, filename, original_room, corrected_room)
            VALUES (datetime('now'), ?, ?, ?)
        ''', (filename, original_room, corrected_room))
        conn.commit()
        conn.close()
        self.increment_stat('total_room_corrections')

    def log_trim_correction(self, filename: str, original_in: float, original_out: float,
                           corrected_in: float, corrected_out: float, total_duration: float):
        """Log a trim point correction for learning optimal in/out points"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trim_corrections (timestamp, filename, original_in, original_out,
                                         corrected_in, corrected_out, total_duration)
            VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)
        ''', (filename, original_in, original_out, corrected_in, corrected_out, total_duration))
        conn.commit()
        conn.close()
        self.increment_stat('total_trim_corrections')

    def log_photo_room_detection(self, filename: str, folder_path: str, detected_room: str,
                                  confidence: float, objects_detected: List[str] = None):
        """Log a photo room detection for cross-training video detection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        objects_str = ",".join(objects_detected) if objects_detected else ""
        cursor.execute('''
            INSERT INTO photo_room_detections (timestamp, filename, folder_path, detected_room, confidence, objects_detected)
            VALUES (datetime('now'), ?, ?, ?, ?, ?)
        ''', (filename, folder_path, detected_room, confidence, objects_str))
        conn.commit()
        conn.close()

    def get_photo_room_hints(self, folder_path: str = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get photo room detections to help train video classification.
        Returns dict of {room_type: [(filename, confidence), ...]}
        If folder_path provided, only returns detections from that folder.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if folder_path:
            cursor.execute('''
                SELECT detected_room, filename, confidence, objects_detected
                FROM photo_room_detections
                WHERE folder_path = ?
                ORDER BY confidence DESC
            ''', (folder_path,))
        else:
            # Get recent detections (last 30 days)
            cursor.execute('''
                SELECT detected_room, filename, confidence, objects_detected
                FROM photo_room_detections
                WHERE timestamp > datetime('now', '-30 days')
                ORDER BY confidence DESC
            ''')

        results = cursor.fetchall()
        conn.close()

        hints = {}
        for room, filename, confidence, objects in results:
            if room not in hints:
                hints[room] = []
            hints[room].append((filename, confidence, objects))

        return hints

    def get_room_object_patterns(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze photo room detections to learn which objects appear in which rooms.
        Returns {room_type: {object: count, ...}, ...}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT detected_room, objects_detected
            FROM photo_room_detections
            WHERE confidence > 0.5 AND objects_detected != ''
        ''')
        results = cursor.fetchall()
        conn.close()

        patterns = {}
        for room, objects_str in results:
            if room not in patterns:
                patterns[room] = {}
            for obj in objects_str.split(','):
                obj = obj.strip()
                if obj:
                    patterns[room][obj] = patterns[room].get(obj, 0) + 1

        return patterns

    def get_trim_correction_patterns(self) -> Dict:
        """Get patterns of trim corrections for learning optimal trim points"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get average adjustment patterns
        cursor.execute('''
            SELECT
                AVG(corrected_in - original_in) as avg_in_adjustment,
                AVG(original_out - corrected_out) as avg_out_adjustment,
                AVG((corrected_out - corrected_in) / total_duration) as avg_hero_ratio,
                COUNT(*) as total_corrections
            FROM trim_corrections
        ''')
        result = cursor.fetchone()
        conn.close()

        if result and result[3] > 0:
            return {
                'avg_in_adjustment': result[0] or 0,
                'avg_out_adjustment': result[1] or 0,
                'avg_hero_ratio': result[2] or 0.8,
                'total_corrections': result[3]
            }
        return {
            'avg_in_adjustment': 0,
            'avg_out_adjustment': 0,
            'avg_hero_ratio': 0.8,
            'total_corrections': 0
        }

    def get_room_correction_patterns(self) -> Dict:
        """Get patterns of room corrections for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT original_room, corrected_room, COUNT(*) as count
            FROM room_corrections
            GROUP BY original_room, corrected_room
            ORDER BY count DESC
        ''')
        results = cursor.fetchall()
        conn.close()
        return [{'from': r[0], 'to': r[1], 'count': r[2]} for r in results]

    def estimate_time_saved(self) -> float:
        """
        Estimate time saved vs manual QA.
        Assumptions:
        - Manual photo QA: ~30 seconds per photo
        - Manual video QA: ~2 minutes per video
        - Manual clip sorting: ~45 seconds per clip
        - Tool reduces this by ~80%
        """
        stats = self.get_all_stats()
        photos = stats.get('total_photos_analyzed', 0)
        videos = stats.get('total_videos_analyzed', 0)
        clips = stats.get('total_clips_analyzed', 0)

        # Manual time in seconds
        manual_time = (photos * 30) + (videos * 120) + (clips * 45)

        # Tool saves ~80% of this time
        time_saved_seconds = manual_time * 0.8

        return time_saved_seconds

    def format_time_saved(self) -> str:
        """Return formatted time saved string"""
        seconds = self.estimate_time_saved()
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


# Global stats tracker instance
stats_tracker = StatsTracker()


# ============================================================================
# FOOTER & AUTHENTICATION
# ============================================================================

def render_footer():
    """Render the stats footer - shown on all pages"""
    all_stats = stats_tracker.get_all_stats()
    total_files = (all_stats.get('total_photos_analyzed', 0) +
                   all_stats.get('total_videos_analyzed', 0) +
                   all_stats.get('total_clips_analyzed', 0))
    total_issues = all_stats.get('total_issues_found', 0)
    time_saved = stats_tracker.format_time_saved()

    st.markdown(f"""
    <div style="border-top: 1px solid #1d1d1f; margin-top: 60px; padding: 30px 20px;">
        <div style="display: flex; justify-content: center; gap: 40px; margin-bottom: 16px;">
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #7B8CDE;">{total_files:,}</div>
                <div style="font-size: 11px; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Files Analyzed</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #7B8CDE;">{total_issues:,}</div>
                <div style="font-size: 11px; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Issues Found</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #7B8CDE;">{time_saved}</div>
                <div style="font-size: 11px; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Time Saved</div>
            </div>
        </div>
        <p style="text-align: center; font-size: 11px; color: #71717a !important; letter-spacing: 0.05em;">Proof by Aerial Canvas · Beta v1.2</p>
    </div>
    """, unsafe_allow_html=True)


def check_authentication():
    """
    Check if user is authenticated via Google OAuth.
    Returns (is_authenticated, user_info) tuple.
    user_info contains: email, name, picture_url
    """
    # TODO: Set to False to enable Google authentication
    AUTH_DISABLED = True  # Temporarily disabled while Google OAuth propagates

    if AUTH_DISABLED:
        return True, {'email': 'team@aerialcanvas.com', 'name': 'Aerial Canvas Team', 'is_team': True}

    if not GOOGLE_AUTH_AVAILABLE:
        # If Google auth not installed, allow access (dev mode)
        return True, {'email': 'dev@aerialcanvas.com', 'name': 'Developer', 'is_team': True}

    # Check if we have valid auth in session
    if 'user_info' in st.session_state and st.session_state.user_info:
        return True, st.session_state.user_info

    return False, None


def show_login_page():
    """Display the Google Sign-In page"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    .stApp, .main, .block-container { background: #000 !important; }
    * { font-family: 'Poppins', -apple-system, sans-serif !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 50vh; text-align: center;">
        <h1 style="font-size: 48px; color: #7B8CDE !important; margin-bottom: 16px;">Proof</h1>
        <p style="color: #a1a1aa; font-size: 18px; margin-bottom: 8px;">by Aerial Canvas</p>
        <p style="color: #71717a; font-size: 14px; margin-bottom: 40px;">Automated QA for video and photo deliverables</p>
    </div>
    """, unsafe_allow_html=True)

    # Google Sign-In
    try:
        client_id = st.secrets["google_oauth"]["client_id"]
        client_secret = st.secrets["google_oauth"]["client_secret"]
        redirect_uri = st.secrets["google_oauth"]["redirect_uri"]

        # Google OAuth endpoints
        AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
        TOKEN_URL = "https://oauth2.googleapis.com/token"

        # Create OAuth2 component
        oauth2 = OAuth2Component(
            client_id=client_id,
            client_secret=client_secret,
            authorize_endpoint=AUTHORIZE_URL,
            token_endpoint=TOKEN_URL,
        )

        # Center the login button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            result = oauth2.authorize_button(
                name="Sign in with Google",
                icon="https://www.google.com/favicon.ico",
                redirect_uri=redirect_uri,
                scope="openid email profile",
                key="google_login",
                use_container_width=True,
            )

        if result and 'token' in result:
            # Get user info from Google
            token = result['token']
            access_token = token.get('access_token')

            if access_token:
                # Fetch user info from Google
                import requests as req
                user_response = req.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"}
                )

                if user_response.status_code == 200:
                    google_user = user_response.json()
                    user_info = {
                        'email': google_user.get('email', ''),
                        'name': google_user.get('name', ''),
                        'picture_url': google_user.get('picture', ''),
                        'is_team': user_db.is_team_member(google_user.get('email', ''))
                    }
                    st.session_state.user_info = user_info

                    # Log to database
                    user_db.get_or_create_user(
                        email=user_info['email'],
                        name=user_info['name'],
                        picture_url=user_info['picture_url']
                    )

                    st.rerun()

    except Exception as e:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <p style="color: #f87171; font-size: 14px;">Google OAuth not configured.</p>
            <p style="color: #71717a; font-size: 12px; margin-top: 8px;">
                See .streamlit/secrets.toml for setup instructions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Dev bypass button (always show for now)
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continue as Developer", use_container_width=True):
            st.session_state.user_info = {
                'email': 'dev@aerialcanvas.com',
                'name': 'Developer',
                'picture_url': '',
                'is_team': True
            }
            st.rerun()


def show_waitlist_page(user_info: dict):
    """Display the waitlist page for non-team users"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    .stApp, .main, .block-container { background: #000 !important; }
    * { font-family: 'Poppins', -apple-system, sans-serif !important; }
    </style>
    """, unsafe_allow_html=True)

    email = user_info.get('email', '')
    name = user_info.get('name', '')

    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 60vh; text-align: center;">
        <h1 style="font-size: 36px; color: #7B8CDE !important; margin-bottom: 16px;">You're on the list!</h1>
        <p style="color: #a1a1aa; font-size: 16px; margin-bottom: 8px;">Thanks for your interest in Proof by Aerial Canvas.</p>
        <p style="color: #71717a; font-size: 14px; margin-bottom: 32px;">
            We're currently in private beta with the Aerial Canvas team.<br>
            We'll notify you at <span style="color: #7B8CDE;">{email}</span> when we open up access.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Add to waitlist
    user_db.add_to_waitlist(email, name)

    # Sign out option
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Sign out", use_container_width=True):
            st.session_state.user_info = None
            st.session_state.connected = False
            st.rerun()


# ============================================================================
# VIDEO ANALYSIS FUNCTIONS
# ============================================================================

def get_video_metadata(file_path: str) -> dict:
    """Extract video metadata using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# ASPECT RATIO & FORMAT DETECTION
# ============================================================================

# Common aspect ratios and their classifications
ASPECT_RATIO_FORMATS = {
    # Vertical (Social Media)
    (9, 16): {'name': '9:16 Vertical', 'type': 'vertical', 'use': 'Social Media (Instagram Reels, TikTok, YouTube Shorts)'},
    (4, 5): {'name': '4:5 Portrait', 'type': 'vertical', 'use': 'Instagram Feed Portrait'},
    (2, 3): {'name': '2:3 Portrait', 'type': 'vertical', 'use': 'Social Media Portrait'},

    # Square
    (1, 1): {'name': '1:1 Square', 'type': 'square', 'use': 'Instagram Feed Square'},

    # Standard Horizontal
    (16, 9): {'name': '16:9 Widescreen', 'type': 'horizontal', 'use': 'YouTube, Standard Video'},
    (17, 9): {'name': '17:9 DCI', 'type': 'horizontal', 'use': 'DCI Standard'},
    (3, 2): {'name': '3:2', 'type': 'horizontal', 'use': 'Photography/Cinema Hybrid'},
    (4, 3): {'name': '4:3', 'type': 'horizontal', 'use': 'Classic TV Format'},

    # Cinematic (Ultra-wide)
    (21, 9): {'name': '21:9 Ultra-wide', 'type': 'cinematic', 'use': 'Cinematic Ultra-wide'},
    (64, 27): {'name': '2.37:1 Scope', 'type': 'cinematic', 'use': 'Anamorphic/CinemaScope'},
    (12, 5): {'name': '2.40:1 Cinematic', 'type': 'cinematic', 'use': 'Modern Anamorphic'},
    (47, 20): {'name': '2.35:1', 'type': 'cinematic', 'use': 'Classic Anamorphic'},
    (239, 100): {'name': '2.39:1', 'type': 'cinematic', 'use': 'Panavision'},
}

# Resolution tiers
RESOLUTION_TIERS = {
    'uhd_8k': {'min_width': 7680, 'label': '8K UHD'},
    'uhd_4k': {'min_width': 3840, 'label': '4K UHD'},
    'qhd': {'min_width': 2560, 'label': '2K/QHD'},
    'fhd': {'min_width': 1920, 'label': '1080p Full HD'},
    'hd': {'min_width': 1280, 'label': '720p HD'},
    'sd': {'min_width': 0, 'label': 'SD'},
}


def detect_video_format(metadata: dict) -> dict:
    """
    Detect video format, aspect ratio, and classification from metadata.

    Returns dict with:
        - width, height: Raw dimensions
        - aspect_ratio: Simplified ratio tuple (e.g., (16, 9))
        - aspect_ratio_decimal: Decimal ratio (e.g., 1.778)
        - aspect_ratio_name: Human-readable name (e.g., "16:9 Widescreen")
        - format_type: 'vertical', 'square', 'horizontal', 'cinematic'
        - is_vertical: Boolean for social media vertical format
        - is_cinematic: Boolean for ultra-wide cinematic format
        - resolution_tier: '4K UHD', '1080p Full HD', etc.
        - use_case: Suggested use (e.g., "Social Media", "YouTube")
        - requirements: Dict of recommended specs for this format
    """
    result = {
        'width': 0,
        'height': 0,
        'aspect_ratio': (16, 9),
        'aspect_ratio_decimal': 1.778,
        'aspect_ratio_name': 'Unknown',
        'format_type': 'horizontal',
        'is_vertical': False,
        'is_cinematic': False,
        'resolution_tier': 'Unknown',
        'use_case': 'Unknown',
        'requirements': {}
    }

    try:
        video_stream = next(s for s in metadata.get('streams', []) if s.get('codec_type') == 'video')
        width = video_stream.get('width', 0)
        height = video_stream.get('height', 0)

        if width == 0 or height == 0:
            return result

        result['width'] = width
        result['height'] = height

        # Calculate aspect ratio
        from math import gcd
        divisor = gcd(width, height)
        ratio_w = width // divisor
        ratio_h = height // divisor
        decimal_ratio = width / height

        result['aspect_ratio'] = (ratio_w, ratio_h)
        result['aspect_ratio_decimal'] = round(decimal_ratio, 3)

        # Determine format type based on ratio
        if decimal_ratio < 0.9:  # Clearly vertical
            result['format_type'] = 'vertical'
            result['is_vertical'] = True
        elif decimal_ratio < 1.1:  # Square-ish
            result['format_type'] = 'square'
        elif decimal_ratio < 2.0:  # Standard horizontal
            result['format_type'] = 'horizontal'
        else:  # Ultra-wide/cinematic
            result['format_type'] = 'cinematic'
            result['is_cinematic'] = True

        # Match to known aspect ratios (with tolerance)
        best_match = None
        best_diff = float('inf')

        for known_ratio, info in ASPECT_RATIO_FORMATS.items():
            known_decimal = known_ratio[0] / known_ratio[1]
            diff = abs(decimal_ratio - known_decimal)
            if diff < best_diff and diff < 0.05:  # 5% tolerance
                best_diff = diff
                best_match = (known_ratio, info)

        if best_match:
            result['aspect_ratio'] = best_match[0]
            result['aspect_ratio_name'] = best_match[1]['name']
            result['format_type'] = best_match[1]['type']
            result['use_case'] = best_match[1]['use']
            result['is_vertical'] = best_match[1]['type'] == 'vertical'
            result['is_cinematic'] = best_match[1]['type'] == 'cinematic'
        else:
            # Custom aspect ratio
            result['aspect_ratio_name'] = f"{ratio_w}:{ratio_h} (Custom)"
            result['use_case'] = 'Custom Format'

        # Determine resolution tier
        # For vertical video, use height as the "width" for tier determination
        effective_width = height if result['is_vertical'] else width

        for tier_name, tier_info in RESOLUTION_TIERS.items():
            if effective_width >= tier_info['min_width']:
                result['resolution_tier'] = tier_info['label']
                break

        # Set format-specific requirements
        if result['is_vertical']:
            result['requirements'] = {
                'recommended_res': '1080x1920',
                'min_res': '720x1280',
                'max_duration': 60,  # Most platforms have 60s limit for shorts
                'use_case': 'Social Media',
                'notes': 'Optimized for Instagram Reels, TikTok, YouTube Shorts'
            }
        elif result['is_cinematic']:
            result['requirements'] = {
                'recommended_res': '4096x1716' if decimal_ratio > 2.3 else '3840x1600',
                'min_res': '1920x800',
                'use_case': 'Cinematic/Film',
                'notes': 'Cinematic aspect ratio - ensure letterboxing is intentional'
            }
        else:
            # Standard horizontal
            result['requirements'] = {
                'recommended_res': '3840x2160',
                'min_res': '1920x1080',
                'use_case': 'YouTube/Standard Delivery',
                'notes': '4K preferred for main deliverables'
            }

        return result

    except Exception as e:
        result['error'] = str(e)
        return result


def check_aspect_ratio(metadata: dict, filename: str = "") -> QAIssue:
    """
    Check video aspect ratio and format classification.
    Returns info about the detected format and any issues.
    """
    format_info = detect_video_format(metadata)

    if format_info.get('error'):
        return QAIssue(
            check_name="Aspect Ratio",
            status="fail",
            message=f"Could not detect aspect ratio: {format_info['error']}"
        )

    width = format_info['width']
    height = format_info['height']
    ratio_name = format_info['aspect_ratio_name']
    format_type = format_info['format_type']
    use_case = format_info['use_case']
    res_tier = format_info['resolution_tier']

    # Build status message
    message_parts = [
        f"{width}x{height}",
        f"({ratio_name})",
        f"· {res_tier}",
        f"· {use_case}"
    ]

    # Check if filename hints don't match detected format
    filename_lower = filename.lower()
    has_social_hint = any(hint in filename_lower for hint in ['sme', 'social', 'reel', 'tiktok', 'short', 'vertical'])
    has_cinematic_hint = any(hint in filename_lower for hint in ['cine', 'film', 'scope', 'anamorphic'])

    status = "pass"
    action = None

    # Validation checks
    if has_social_hint and not format_info['is_vertical']:
        status = "warning"
        action = f"Filename suggests social media but video is {format_type}. Verify correct export."
    elif has_cinematic_hint and not format_info['is_cinematic']:
        status = "warning"
        action = f"Filename suggests cinematic but video is {format_type}. Verify correct export."
    elif format_info['is_vertical'] and not has_social_hint:
        status = "info"
        message_parts.append("(Auto-detected as social media content)")

    # Check for unusual/custom ratios
    if 'Custom' in ratio_name:
        status = "info"
        action = "Non-standard aspect ratio detected. Verify this is intentional."

    return QAIssue(
        check_name="Aspect Ratio",
        status=status,
        message=" ".join(message_parts),
        action=action
    )


def check_video_resolution(metadata: dict, is_vertical: bool = False, format_info: dict = None) -> QAIssue:
    """Check if video resolution meets requirements based on detected format"""
    try:
        video_stream = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
        width = video_stream['width']
        height = video_stream['height']

        # Use format_info if provided, otherwise fall back to is_vertical flag
        if format_info is None:
            format_info = detect_video_format(metadata)

        format_type = format_info.get('format_type', 'horizontal')
        is_vertical = format_info.get('is_vertical', is_vertical)
        is_cinematic = format_info.get('is_cinematic', False)
        ratio_name = format_info.get('aspect_ratio_name', '')
        res_tier = format_info.get('resolution_tier', '')

        # VERTICAL (Social Media) - 9:16
        if is_vertical:
            if width >= 1080 and height >= 1920:
                return QAIssue(
                    check_name="Resolution",
                    status="pass",
                    message=f"{width}x{height} ({ratio_name} · {res_tier})"
                )
            elif width >= 720 and height >= 1280:
                return QAIssue(
                    check_name="Resolution",
                    status="warning",
                    message=f"{width}x{height} - Acceptable but not optimal for social",
                    expected="1080x1920 recommended",
                    found=f"{width}x{height}",
                    action="Consider re-exporting at 1080x1920 for best quality"
                )
            else:
                return QAIssue(
                    check_name="Resolution",
                    status="fail",
                    message=f"Resolution too low for social media",
                    expected="1080x1920 minimum",
                    found=f"{width}x{height}",
                    action="Re-export at 1080x1920 for vertical/SME"
                )

        # CINEMATIC (Ultra-wide: 2.35:1, 2.40:1, etc.)
        elif is_cinematic:
            # For cinematic, check the longer dimension (width)
            if width >= 3840:
                return QAIssue(
                    check_name="Resolution",
                    status="pass",
                    message=f"{width}x{height} ({ratio_name} · 4K Cinematic)"
                )
            elif width >= 1920:
                return QAIssue(
                    check_name="Resolution",
                    status="pass",
                    message=f"{width}x{height} ({ratio_name} · HD Cinematic)"
                )
            else:
                return QAIssue(
                    check_name="Resolution",
                    status="warning",
                    message=f"Cinematic resolution may be too low",
                    expected="3840+ width for 4K cinematic",
                    found=f"{width}x{height}",
                    action="Verify resolution is intentional for delivery platform"
                )

        # SQUARE (1:1)
        elif format_type == 'square':
            if width >= 1080 and height >= 1080:
                return QAIssue(
                    check_name="Resolution",
                    status="pass",
                    message=f"{width}x{height} (1:1 Square · {res_tier})"
                )
            else:
                return QAIssue(
                    check_name="Resolution",
                    status="warning",
                    message=f"Square video resolution is low",
                    expected="1080x1080 minimum",
                    found=f"{width}x{height}",
                    action="Consider re-exporting at higher resolution"
                )

        # STANDARD HORIZONTAL (16:9, 17:9, etc.)
        else:
            if width >= 3840 and height >= 2160:
                return QAIssue(
                    check_name="Resolution",
                    status="pass",
                    message=f"{width}x{height} ({ratio_name} · 4K)"
                )
            elif width >= 1920 and height >= 1080:
                # 1080p is acceptable but 4K preferred for main deliverables
                return QAIssue(
                    check_name="Resolution",
                    status="warning",
                    message=f"{width}x{height} ({ratio_name} · 1080p)",
                    expected="3840x2160 (4K) preferred",
                    found=f"{width}x{height}",
                    action="4K (3840x2160) recommended for main deliverables"
                )
            else:
                return QAIssue(
                    check_name="Resolution",
                    status="fail",
                    message=f"Resolution below HD standard",
                    expected="1920x1080 minimum",
                    found=f"{width}x{height}",
                    action="Re-export at minimum 1080p, preferably 4K"
                )
    except Exception as e:
        return QAIssue(
            check_name="Resolution",
            status="fail",
            message=f"Could not read resolution: {str(e)}"
        )

def check_frame_rate(metadata: dict) -> QAIssue:
    """Check video frame rate"""
    try:
        video_stream = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
        fps_str = video_stream.get('r_frame_rate', '0/1')
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den > 0 else 0

        if 23.9 <= fps <= 24.1:
            return QAIssue(check_name="Frame Rate", status="pass", message=f"{fps:.2f} fps (24p)")
        elif 29.9 <= fps <= 30.1:
            return QAIssue(check_name="Frame Rate", status="pass", message=f"{fps:.2f} fps (30p)")
        elif 59.9 <= fps <= 60.1:
            return QAIssue(check_name="Frame Rate", status="pass", message=f"{fps:.2f} fps (60p)")
        else:
            return QAIssue(
                check_name="Frame Rate",
                status="warning",
                message=f"Unusual frame rate: {fps:.2f} fps",
                action="Verify frame rate is intentional"
            )
    except Exception as e:
        return QAIssue(check_name="Frame Rate", status="fail", message=f"Could not read frame rate: {str(e)}")

def check_audio_present(metadata: dict) -> QAIssue:
    """Check if audio stream exists"""
    try:
        audio_streams = [s for s in metadata['streams'] if s['codec_type'] == 'audio']
        if audio_streams:
            audio = audio_streams[0]
            channels = audio.get('channels', 'unknown')
            sample_rate = audio.get('sample_rate', 'unknown')
            return QAIssue(
                check_name="Audio",
                status="pass",
                message=f"Audio present ({channels}ch, {sample_rate}Hz)"
            )
        else:
            return QAIssue(
                check_name="Audio",
                status="fail",
                message="No audio track found",
                action="Add audio/music track to video"
            )
    except Exception as e:
        return QAIssue(check_name="Audio", status="fail", message=f"Could not check audio: {str(e)}")

def detect_black_frames(file_path: str, duration: float, sample_only: bool = False) -> QAIssue:
    """Detect black frames in video

    Args:
        file_path: Path to video file
        duration: Video duration in seconds
        sample_only: If True, only sample beginning, middle, and end (faster for quick mode)
    """
    try:
        if sample_only and duration > 30:
            # Quick mode: Only analyze first 10s, middle 10s, and last 10s
            black_frames = []
            segments = [
                (0, min(10, duration)),  # First 10 seconds
                (max(0, duration/2 - 5), min(duration, duration/2 + 5)),  # Middle 10 seconds
                (max(0, duration - 10), duration)  # Last 10 seconds
            ]

            for start_time, end_time in segments:
                cmd = [
                    'ffmpeg', '-ss', str(start_time), '-i', file_path,
                    '-t', str(end_time - start_time),
                    '-vf', 'blackdetect=d=0.1:pix_th=0.10',
                    '-an', '-f', 'null', '-'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                for line in result.stderr.split('\n'):
                    if 'black_start' in line:
                        match = re.search(r'black_start:([\d.]+)\s+black_end:([\d.]+)', line)
                        if match:
                            # Adjust timestamps to absolute position
                            start = float(match.group(1)) + start_time
                            end = float(match.group(2)) + start_time
                            black_frames.append((start, end))
        else:
            # Full analysis
            cmd = [
                'ffmpeg', '-i', file_path,
                '-vf', 'blackdetect=d=0.1:pix_th=0.10',
                '-an', '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            black_frames = []
            for line in result.stderr.split('\n'):
                if 'black_start' in line:
                    match = re.search(r'black_start:([\d.]+)\s+black_end:([\d.]+)', line)
                    if match:
                        start = float(match.group(1))
                        end = float(match.group(2))
                        black_frames.append((start, end))

        if not black_frames:
            return QAIssue(
                check_name="Black Frames",
                status="pass",
                message="No black frame gaps detected"
            )

        # Separate fade out (last 1 second) from problematic gaps
        issues = []
        fade_detected = False

        for start, end in black_frames:
            # If black frames are in the last 1.5 seconds, it's likely fade out
            if start >= duration - 1.5:
                fade_detected = True
            else:
                issues.append((start, end))

        if not issues:
            return QAIssue(
                check_name="Black Frames",
                status="pass",
                message="No black frame gaps detected (fade out at end is normal)"
            )

        # Format issue report
        timestamps = []
        for start, end in issues:
            start_fmt = f"{int(start//60)}:{start%60:05.2f}"
            end_fmt = f"{int(end//60)}:{end%60:05.2f}"
            timestamps.append(f"{start_fmt} - {end_fmt}")

        # Capture a preview frame at the first black frame
        preview = None
        if issues:
            # Capture frame slightly before the black frame to show context
            preview_time = max(0, issues[0][0] - 0.5)
            preview = capture_video_frame(file_path, preview_time)

        return QAIssue(
            check_name="Black Frames",
            status="fail",
            message=f"Black frame gaps detected ({len(issues)} found)",
            timestamp_start=issues[0][0] if issues else None,
            action="Check edit timeline for gaps at: " + ", ".join(timestamps),
            preview_image=preview
        )

    except subprocess.TimeoutExpired:
        return QAIssue(check_name="Black Frames", status="warning", message="Analysis timed out")
    except Exception as e:
        return QAIssue(check_name="Black Frames", status="fail", message=f"Error: {str(e)}")

def detect_log_footage(file_path: str, duration: float) -> QAIssue:
    """Detect ungraded/log footage by analyzing contrast and saturation"""
    try:
        # Sample frames throughout the video
        sample_times = [duration * i / 10 for i in range(1, 10)]
        low_contrast_segments = []

        for t in sample_times:
            # Extract a frame and analyze it
            cmd = [
                'ffmpeg', '-ss', str(t), '-i', file_path,
                '-vframes', '1', '-f', 'image2pipe',
                '-vcodec', 'png', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode == 0 and result.stdout:
                # Analyze the frame for contrast/saturation
                # Low contrast + low saturation = likely log footage
                import cv2
                import numpy as np

                # Decode image from bytes
                nparr = np.frombuffer(result.stdout, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    # Convert to different color spaces for analysis
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                    # Calculate contrast (standard deviation of luminance)
                    contrast = np.std(gray)

                    # Calculate saturation
                    saturation = np.mean(hsv[:, :, 1])

                    # Log footage detection thresholds (calibrated from actual AC examples)
                    # Log footage averages: contrast ~29, saturation ~17
                    # Graded footage averages: contrast ~58, saturation ~59
                    # Using conservative thresholds to avoid false positives
                    if contrast < 38 and saturation < 30:
                        low_contrast_segments.append({
                            'time': t,
                            'contrast': contrast,
                            'saturation': saturation
                        })

        if not low_contrast_segments:
            return QAIssue(
                check_name="Log Footage Detection",
                status="pass",
                message="All footage appears to be color graded"
            )

        # Format timestamps
        timestamps = []
        for seg in low_contrast_segments:
            t = seg['time']
            time_fmt = f"{int(t//60)}:{t%60:05.2f}"
            timestamps.append(f"{time_fmt}")

        # Capture preview at first log footage location
        preview = capture_video_frame(file_path, low_contrast_segments[0]['time'])

        return QAIssue(
            check_name="Log Footage Detection",
            status="fail",
            message=f"Potential ungraded footage detected ({len(low_contrast_segments)} locations)",
            timestamp_start=low_contrast_segments[0]['time'],
            action="Apply color grade at: " + ", ".join(timestamps),
            preview_image=preview
        )

    except ImportError:
        return QAIssue(
            check_name="Log Footage Detection",
            status="warning",
            message="OpenCV not installed - skipping log detection"
        )
    except Exception as e:
        return QAIssue(
            check_name="Log Footage Detection",
            status="warning",
            message=f"Could not analyze: {str(e)}"
        )

def check_fade_out(file_path: str, duration: float) -> QAIssue:
    """Check if video has proper fade out"""
    try:
        # Check the last 2 seconds for decreasing brightness
        cmd = [
            'ffmpeg', '-ss', str(max(0, duration - 2)), '-i', file_path,
            '-vf', 'blackdetect=d=0.05:pix_th=0.15',
            '-an', '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if 'black_start' in result.stderr:
            return QAIssue(
                check_name="Fade Out",
                status="pass",
                message="Fade to black detected at end"
            )
        else:
            # Capture preview of the last frame to show abrupt ending
            preview = capture_video_frame(file_path, max(0, duration - 0.5))
            return QAIssue(
                check_name="Fade Out",
                status="warning",
                message="No fade out detected - video may end abruptly",
                timestamp_start=duration,  # Mark at end of timeline
                action="Add fade out to end of video",
                preview_image=preview
            )
    except Exception as e:
        return QAIssue(check_name="Fade Out", status="warning", message=f"Could not check: {str(e)}")

def check_stabilization(file_path: str, duration: float) -> QAIssue:
    """
    Check for vertical bobbing/jogging motion specifically.
    Real estate videos often have intentional fast lateral movement (gimbal/drone),
    but vertical bobbing from walking/jogging is problematic.
    Only flags the worst segments, not all shake.
    """
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return QAIssue(check_name="Stabilization", status="warning", message="Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames throughout the video (every 0.3 seconds for better detection)
        sample_interval = int(fps * 0.3) if fps > 0 else 10
        vertical_motions = []  # Track vertical (Y) motion specifically
        bobbing_segments = []

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return QAIssue(check_name="Stabilization", status="warning", message="Could not read frames")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_idx = 0

        # Track consecutive vertical motion (bobbing pattern detection)
        vertical_history = []

        while True:
            # Skip frames to sample interval
            for _ in range(sample_interval - 1):
                ret = cap.grab()
                if not ret:
                    break
                frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Separate horizontal (X) and vertical (Y) motion
            horizontal_flow = flow[..., 0]
            vertical_flow = flow[..., 1]  # Y-axis motion (up-down)

            # Calculate mean vertical motion (positive = down, negative = up)
            mean_vertical = np.mean(vertical_flow)
            vertical_variance = np.std(vertical_flow)

            # Track vertical motion history for bobbing pattern detection
            vertical_history.append(mean_vertical)
            if len(vertical_history) > 6:  # Keep last 6 samples (~2 seconds)
                vertical_history.pop(0)

            # Detect bobbing: look for oscillating up-down pattern
            # Bobbing = significant vertical motion with direction changes
            is_bobbing = False
            if len(vertical_history) >= 4:
                # Check for direction changes (sign changes in vertical motion)
                signs = [1 if v > 0.5 else (-1 if v < -0.5 else 0) for v in vertical_history]
                sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != 0 and signs[i-1] != 0 and signs[i] != signs[i-1])

                # Also check magnitude of vertical motion
                max_vertical = max(abs(v) for v in vertical_history)

                # Bobbing: multiple direction changes + significant vertical movement
                if sign_changes >= 2 and max_vertical > 3.0:
                    is_bobbing = True

            vertical_motions.append({
                'time': frame_idx / fps if fps > 0 else 0,
                'vertical_var': vertical_variance,
                'mean_vertical': abs(mean_vertical),
                'is_bobbing': is_bobbing
            })

            # Only flag significant vertical bobbing (not horizontal pan/tilt)
            if is_bobbing and vertical_variance > 4.0:
                timestamp = frame_idx / fps if fps > 0 else 0
                bobbing_segments.append({
                    'time': timestamp,
                    'severity': vertical_variance,
                    'type': 'bobbing'
                })

            prev_gray = gray

        cap.release()

        if not vertical_motions:
            return QAIssue(check_name="Stabilization", status="warning", message="Could not analyze motion")

        # Calculate overall vertical motion stats
        avg_vertical_var = np.mean([m['vertical_var'] for m in vertical_motions])
        bobbing_count = sum(1 for m in vertical_motions if m['is_bobbing'])

        # Only report if significant bobbing detected
        if not bobbing_segments:
            return QAIssue(
                check_name="Stabilization",
                status="pass",
                message=f"Smooth footage - no vertical bobbing detected"
            )

        # Find the worst bobbing segment (not all of them)
        worst_segment = max(bobbing_segments, key=lambda x: x['severity'])

        # Capture preview at worst bobbing location
        preview = capture_video_frame(file_path, worst_segment['time'])

        # Determine severity based on how many bobbing segments
        if len(bobbing_segments) <= 2:
            return QAIssue(
                check_name="Stabilization",
                status="warning",
                message=f"Minor vertical bobbing detected",
                timestamp_start=worst_segment['time'],
                action="Review footage for walking/jogging motion",
                preview_image=preview
            )
        else:
            return QAIssue(
                check_name="Stabilization",
                status="fail",
                message=f"Vertical bobbing detected ({len(bobbing_segments)} locations)",
                timestamp_start=worst_segment['time'],
                action="Stabilize or replace jogging/walking footage",
                preview_image=preview
            )

    except ImportError:
        return QAIssue(check_name="Stabilization", status="warning", message="OpenCV not installed")
    except Exception as e:
        return QAIssue(check_name="Stabilization", status="warning", message=f"Could not analyze: {str(e)}")

def detect_scene_cuts(file_path: str, duration: float) -> list:
    """
    Detect scene cuts/transitions in the video.
    Returns list of (timestamp, cut_strength) tuples.
    """
    try:
        # Use ffmpeg's scene detection
        cmd = [
            'ffmpeg', '-i', file_path,
            '-vf', 'select=\'gt(scene,0.3)\',showinfo',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        cuts = []
        for line in result.stderr.split('\n'):
            if 'pts_time' in line:
                # Extract timestamp
                match = re.search(r'pts_time:([\d.]+)', line)
                if match:
                    timestamp = float(match.group(1))
                    cuts.append(timestamp)

        return cuts
    except:
        return []

def check_transition_smoothness(file_path: str, duration: float) -> QAIssue:
    """
    Check if transitions between shots are smooth vs jarring.
    Analyzes scene cuts for abruptness.
    """
    try:
        import cv2
        import numpy as np

        # Get scene cuts
        cuts = detect_scene_cuts(file_path, duration)

        if not cuts:
            return QAIssue(
                check_name="Transitions",
                status="pass",
                message="No harsh cuts detected"
            )

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        jarring_cuts = []

        for cut_time in cuts:
            if cut_time < 0.5 or cut_time > duration - 0.5:
                continue

            # Get frames before and after cut
            cap.set(cv2.CAP_PROP_POS_MSEC, (cut_time - 0.1) * 1000)
            ret1, frame_before = cap.read()

            cap.set(cv2.CAP_PROP_POS_MSEC, (cut_time + 0.1) * 1000)
            ret2, frame_after = cap.read()

            if ret1 and ret2:
                # Calculate histogram difference
                hist_before = cv2.calcHist([frame_before], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist_after = cv2.calcHist([frame_after], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

                hist_before = cv2.normalize(hist_before, hist_before).flatten()
                hist_after = cv2.normalize(hist_after, hist_after).flatten()

                # Compare histograms (lower = more similar)
                diff = cv2.compareHist(hist_before, hist_after, cv2.HISTCMP_CHISQR)

                # Very high difference indicates jarring cut
                if diff > 50:
                    jarring_cuts.append({
                        'time': cut_time,
                        'severity': diff
                    })

        cap.release()

        if not jarring_cuts:
            return QAIssue(
                check_name="Transitions",
                status="pass",
                message=f"All {len(cuts)} cuts are smooth"
            )

        # Get the worst (most jarring) cut for the timestamp marker
        worst_cut = max(jarring_cuts, key=lambda x: x['severity'])
        preview = capture_video_frame(file_path, worst_cut['time'])

        if len(jarring_cuts) <= 2:
            timestamps = [f"{int(c['time']//60)}:{c['time']%60:05.2f}" for c in jarring_cuts]
            return QAIssue(
                check_name="Transitions",
                status="warning",
                message=f"{len(jarring_cuts)} slightly abrupt cut(s)",
                timestamp_start=worst_cut['time'],
                action="Review these transitions: " + ", ".join(timestamps),
                preview_image=preview
            )
        else:
            timestamps = [f"{int(c['time']//60)}:{c['time']%60:05.2f}" for c in jarring_cuts[:5]]
            return QAIssue(
                check_name="Transitions",
                status="fail",
                message=f"{len(jarring_cuts)} jarring cuts detected",
                timestamp_start=worst_cut['time'],
                action="Add transitions or improve edit flow: " + ", ".join(timestamps),
                preview_image=preview
            )

    except Exception as e:
        return QAIssue(check_name="Transitions", status="warning", message=f"Could not analyze: {str(e)}")

def check_beat_sync(file_path: str, duration: float) -> QAIssue:
    """
    Check if video cuts sync with music beats.
    Extracts audio, detects beats, compares to scene cuts.
    """
    try:
        import numpy as np

        # Extract audio to analyze
        audio_path = file_path + '_temp_audio.wav'
        cmd = [
            'ffmpeg', '-y', '-i', file_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
            audio_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)

        # Read audio file
        import wave
        with wave.open(audio_path, 'rb') as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_data = np.frombuffer(wav.readframes(n_frames), dtype=np.int16).astype(np.float32)

        # Clean up temp file
        os.unlink(audio_path)

        # Simple beat detection using onset strength
        # Calculate short-time energy
        frame_length = int(sample_rate * 0.02)  # 20ms frames
        hop_length = int(sample_rate * 0.01)    # 10ms hop

        energy = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            energy.append(np.sum(frame ** 2))

        energy = np.array(energy)

        # Find peaks in energy (potential beats)
        energy_diff = np.diff(energy)
        energy_diff = np.maximum(energy_diff, 0)

        # Normalize
        if np.max(energy_diff) > 0:
            energy_diff = energy_diff / np.max(energy_diff)

        # Find beat times (peaks above threshold)
        beat_threshold = 0.3
        beat_frames = np.where(energy_diff > beat_threshold)[0]

        # Convert to timestamps
        beats = beat_frames * hop_length / sample_rate

        # Reduce to significant beats (at least 0.25s apart)
        filtered_beats = []
        last_beat = -1
        for b in beats:
            if b - last_beat > 0.25:
                filtered_beats.append(b)
                last_beat = b

        beats = filtered_beats

        if len(beats) < 4:
            return QAIssue(
                check_name="Beat Sync",
                status="warning",
                message="Could not detect clear beat pattern in music"
            )

        # Get scene cuts
        cuts = detect_scene_cuts(file_path, duration)

        if not cuts:
            return QAIssue(
                check_name="Beat Sync",
                status="warning",
                message="No cuts detected to sync check"
            )

        # Check how many cuts are near beats (within 0.15s tolerance)
        synced_cuts = 0
        off_beat_cuts = []
        tolerance = 0.15

        for cut in cuts:
            # Find nearest beat
            min_dist = min(abs(cut - b) for b in beats) if beats else float('inf')
            if min_dist <= tolerance:
                synced_cuts += 1
            else:
                off_beat_cuts.append(cut)

        sync_percentage = (synced_cuts / len(cuts)) * 100 if cuts else 0

        if sync_percentage >= 80:
            return QAIssue(
                check_name="Beat Sync",
                status="pass",
                message=f"Great rhythm! {sync_percentage:.0f}% of cuts sync with music"
            )
        elif sync_percentage >= 50:
            timestamps = [f"⏱️ {int(t//60)}:{t%60:05.2f}" for t in off_beat_cuts[:3]]
            return QAIssue(
                check_name="Beat Sync",
                status="warning",
                message=f"{sync_percentage:.0f}% cuts on beat - could be tighter",
                action="Consider adjusting these cuts to hit beats:\n" + "\n".join(timestamps)
            )
        else:
            timestamps = [f"⏱️ {int(t//60)}:{t%60:05.2f}" for t in off_beat_cuts[:5]]
            return QAIssue(
                check_name="Beat Sync",
                status="fail",
                message=f"Only {sync_percentage:.0f}% of cuts sync with music",
                action="Re-edit to cut on the beat:\n" + "\n".join(timestamps)
            )

    except Exception as e:
        return QAIssue(check_name="Beat Sync", status="warning", message=f"Could not analyze: {str(e)}")

def check_audio_levels(file_path: str) -> QAIssue:
    """
    Check audio levels for peaking (too loud/clipping) or too quiet.
    Uses ffmpeg's volumedetect filter.
    """
    try:
        cmd = [
            'ffmpeg', '-i', file_path,
            '-af', 'volumedetect',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Parse volume info from stderr
        max_volume = None
        mean_volume = None

        for line in result.stderr.split('\n'):
            if 'max_volume' in line:
                match = re.search(r'max_volume:\s*([-\d.]+)\s*dB', line)
                if match:
                    max_volume = float(match.group(1))
            if 'mean_volume' in line:
                match = re.search(r'mean_volume:\s*([-\d.]+)\s*dB', line)
                if match:
                    mean_volume = float(match.group(1))

        if max_volume is None:
            return QAIssue(check_name="Audio Levels", status="warning", message="Could not analyze audio levels")

        issues = []

        # Check for clipping (max volume at or above 0 dB)
        if max_volume >= -0.5:
            issues.append(f"CLIPPING detected (peak: {max_volume:.1f} dB)")
        elif max_volume >= -3:
            issues.append(f"Peaks near clipping ({max_volume:.1f} dB)")

        # Check for audio too quiet
        if mean_volume < -30:
            issues.append(f"Audio too QUIET (avg: {mean_volume:.1f} dB)")
        elif mean_volume < -24:
            issues.append(f"Audio slightly quiet (avg: {mean_volume:.1f} dB)")

        # Check for audio too loud overall
        if mean_volume > -10:
            issues.append(f"Audio may be too HOT (avg: {mean_volume:.1f} dB)")

        if not issues:
            return QAIssue(
                check_name="Audio Levels",
                status="pass",
                message=f"Good levels (peak: {max_volume:.1f} dB, avg: {mean_volume:.1f} dB)"
            )
        elif max_volume >= -0.5 or mean_volume < -30:
            return QAIssue(
                check_name="Audio Levels",
                status="fail",
                message="; ".join(issues),
                action="Adjust audio levels - aim for peaks around -6dB, average around -18dB"
            )
        else:
            return QAIssue(
                check_name="Audio Levels",
                status="warning",
                message="; ".join(issues),
                action="Consider normalizing audio levels"
            )

    except Exception as e:
        return QAIssue(check_name="Audio Levels", status="warning", message=f"Could not analyze: {str(e)}")


def check_audio_fade_in_out(file_path: str, duration: float) -> QAIssue:
    """
    Check if audio/music has proper fade in at the start and fade out at the end.
    Analyzes the audio waveform in the first and last few seconds.
    """
    try:
        # Analyze first 2 seconds for fade in
        fade_in_duration = min(2.0, duration * 0.1)
        cmd_in = [
            'ffmpeg', '-i', file_path,
            '-t', str(fade_in_duration),
            '-af', 'volumedetect',
            '-f', 'null', '-'
        ]
        result_in = subprocess.run(cmd_in, capture_output=True, text=True, timeout=60)

        # Analyze last 2 seconds for fade out
        fade_out_start = max(0, duration - 2.0)
        cmd_out = [
            'ffmpeg', '-i', file_path,
            '-ss', str(fade_out_start),
            '-af', 'volumedetect',
            '-f', 'null', '-'
        ]
        result_out = subprocess.run(cmd_out, capture_output=True, text=True, timeout=60)

        # Parse volume levels
        def get_max_volume(stderr):
            for line in stderr.split('\n'):
                if 'max_volume' in line:
                    match = re.search(r'max_volume:\s*([-\d.]+)\s*dB', line)
                    if match:
                        return float(match.group(1))
            return None

        start_vol = get_max_volume(result_in.stderr)
        end_vol = get_max_volume(result_out.stderr)

        # Also get middle section volume for comparison
        mid_start = duration * 0.4
        cmd_mid = [
            'ffmpeg', '-i', file_path,
            '-ss', str(mid_start),
            '-t', '2',
            '-af', 'volumedetect',
            '-f', 'null', '-'
        ]
        result_mid = subprocess.run(cmd_mid, capture_output=True, text=True, timeout=60)
        mid_vol = get_max_volume(result_mid.stderr)

        if start_vol is None or end_vol is None or mid_vol is None:
            return QAIssue(check_name="Audio Fades", status="warning", message="Could not analyze audio fades")

        issues = []

        # Check fade in - start should be quieter than middle
        # A good fade in has at least 6dB difference
        if mid_vol is not None and start_vol is not None:
            fade_in_diff = mid_vol - start_vol
            if fade_in_diff < 3 and start_vol > -30:  # Start is almost as loud as middle
                issues.append("No fade IN on music")

        # Check fade out - end should be quieter than middle
        if mid_vol is not None and end_vol is not None:
            fade_out_diff = mid_vol - end_vol
            if fade_out_diff < 3 and end_vol > -30:  # End is almost as loud as middle
                issues.append("No fade OUT on music")

        if not issues:
            return QAIssue(
                check_name="Audio Fades",
                status="pass",
                message="Music has proper fade in/out"
            )
        else:
            return QAIssue(
                check_name="Audio Fades",
                status="fail",
                message="; ".join(issues),
                action="Add fade in at start and fade out at end of music track"
            )

    except Exception as e:
        return QAIssue(check_name="Audio Fades", status="warning", message=f"Could not analyze: {str(e)}")


def check_audio_noise(file_path: str) -> QAIssue:
    """
    Check for audio noise, fuzziness, or hiss in the audio track.
    Uses FFmpeg's astats filter to detect noise floor.
    """
    try:
        # Use astats to get detailed audio statistics
        cmd = [
            'ffmpeg', '-i', file_path,
            '-af', 'astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.Noise_floor',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Alternative: analyze high frequency content (noise is often high freq)
        # Use highpass filter and measure volume
        cmd_hf = [
            'ffmpeg', '-i', file_path,
            '-af', 'highpass=f=8000,volumedetect',
            '-f', 'null', '-'
        ]
        result_hf = subprocess.run(cmd_hf, capture_output=True, text=True, timeout=300)

        # Parse high frequency volume
        hf_mean = None
        for line in result_hf.stderr.split('\n'):
            if 'mean_volume' in line:
                match = re.search(r'mean_volume:\s*([-\d.]+)\s*dB', line)
                if match:
                    hf_mean = float(match.group(1))

        # Also check for DC offset and other issues with silencedetect
        cmd_silence = [
            'ffmpeg', '-i', file_path,
            '-af', 'silencedetect=n=-50dB:d=0.5',
            '-f', 'null', '-'
        ]
        result_silence = subprocess.run(cmd_silence, capture_output=True, text=True, timeout=300)

        # Count silence periods (too few might indicate constant noise)
        silence_count = result_silence.stderr.count('silence_end')

        issues = []

        # High frequency noise check
        if hf_mean is not None:
            if hf_mean > -35:
                issues.append(f"High frequency noise/hiss detected ({hf_mean:.1f} dB)")
            elif hf_mean > -45:
                issues.append(f"Slight high frequency noise ({hf_mean:.1f} dB)")

        # If there's almost no silence in the audio, might indicate noise floor issues
        # (This is a heuristic - real estate videos with constant music won't have silence)

        if not issues:
            return QAIssue(
                check_name="Audio Noise",
                status="pass",
                message="Audio is clean, no significant noise detected"
            )
        elif hf_mean and hf_mean > -35:
            return QAIssue(
                check_name="Audio Noise",
                status="fail",
                message="; ".join(issues),
                action="Audio has noise/hiss - apply noise reduction or re-export"
            )
        else:
            return QAIssue(
                check_name="Audio Noise",
                status="warning",
                message="; ".join(issues),
                action="Consider applying light noise reduction"
            )

    except Exception as e:
        return QAIssue(check_name="Audio Noise", status="warning", message=f"Could not analyze: {str(e)}")


def check_color_consistency(file_path: str, duration: float) -> QAIssue:
    """
    Check for color consistency across different shots in the video.
    Flags if some shots appear to have different color grading than others.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return QAIssue(check_name="Color Consistency", status="warning", message="OpenCV not available")

    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return QAIssue(check_name="Color Consistency", status="warning", message="Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames throughout the video (every 2 seconds)
        sample_interval = int(fps * 2)
        color_stats = []

        frame_idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            # Calculate color statistics
            h_mean, s_mean, v_mean = cv2.mean(hsv)[:3]
            l_mean, a_mean, b_mean = cv2.mean(lab)[:3]

            # Also calculate warmth (red-blue balance)
            b_channel, g_channel, r_channel = cv2.split(frame)
            warmth = np.mean(r_channel.astype(float) - b_channel.astype(float))

            color_stats.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'hue': h_mean,
                'saturation': s_mean,
                'value': v_mean,
                'lightness': l_mean,
                'warmth': warmth
            })

            frame_idx += sample_interval
            if frame_idx >= total_frames:
                break

        cap.release()

        if len(color_stats) < 3:
            return QAIssue(check_name="Color Consistency", status="pass", message="Video too short for consistency check")

        # Analyze consistency
        saturations = [s['saturation'] for s in color_stats]
        warmths = [s['warmth'] for s in color_stats]
        lightnesses = [s['lightness'] for s in color_stats]

        sat_std = np.std(saturations)
        warmth_std = np.std(warmths)
        light_std = np.std(lightnesses)

        issues = []
        timestamps = []

        # Check for inconsistent saturation (some shots more saturated than others)
        if sat_std > 25:
            issues.append(f"Inconsistent saturation across shots (variance: {sat_std:.1f})")
            # Find the outlier shots
            sat_mean = np.mean(saturations)
            for stat in color_stats:
                if abs(stat['saturation'] - sat_mean) > 30:
                    timestamps.append(f"{int(stat['time']//60)}:{stat['time']%60:05.2f}")

        # Check for inconsistent color temperature (some shots warmer/cooler)
        if warmth_std > 20:
            issues.append(f"Inconsistent color temperature (some shots warmer/cooler)")
            warmth_mean = np.mean(warmths)
            for stat in color_stats:
                if abs(stat['warmth'] - warmth_mean) > 25:
                    if stat['time'] not in [float(t.replace(':', '.')) for t in timestamps]:
                        timestamps.append(f"{int(stat['time']//60)}:{stat['time']%60:05.2f}")

        # Check for inconsistent exposure/brightness
        if light_std > 30:
            issues.append(f"Inconsistent exposure between shots")

        if not issues:
            return QAIssue(
                check_name="Color Consistency",
                status="pass",
                message="Color grading is consistent across shots"
            )
        else:
            msg = "; ".join(issues)
            if timestamps:
                msg += f" | Check around: {', '.join(timestamps[:3])}"
            return QAIssue(
                check_name="Color Consistency",
                status="fail" if sat_std > 35 or warmth_std > 30 else "warning",
                message=msg,
                action="Review color grading - ensure all shots are balanced consistently"
            )

    except Exception as e:
        return QAIssue(check_name="Color Consistency", status="warning", message=f"Could not analyze: {str(e)}")


def check_filename_convention(filename: str, folder_path: str = "") -> QAIssue:
    """
    Check if the filename follows Aerial Canvas naming conventions.
    Expected patterns:
    - YouTube: "Address - Client Name - Agent Brokerage (Product Type).mp4"
    - Dropbox: Similar structure with proper formatting
    """
    issues = []

    # Remove extension for analysis
    name_no_ext = os.path.splitext(filename)[0]

    # Check for common naming issues

    # 1. Should not have underscores (use spaces or hyphens)
    if '_' in name_no_ext and ' - ' not in name_no_ext:
        issues.append("Uses underscores instead of proper formatting")

    # 2. Should have proper separators (typically " - " between sections)
    if ' - ' not in name_no_ext:
        issues.append("Missing proper section separators (use ' - ' between address, client, etc.)")

    # 3. Should not be all lowercase
    if name_no_ext == name_no_ext.lower():
        issues.append("Filename is all lowercase - should use proper capitalization")

    # 4. Should not have special characters except hyphens and parentheses
    invalid_chars = re.findall(r'[^\w\s\-\(\)\.,]', name_no_ext)
    if invalid_chars:
        issues.append(f"Contains invalid characters: {set(invalid_chars)}")

    # 5. Check for common patterns that indicate a good filename
    good_patterns = [
        r'\d+\s+\w+',  # Address number and street
        r'\b(CA|TX|FL|NY|AZ|CO|WA|OR|NV)\b',  # State abbreviation
        r'\b(Video|Photo|Drone|Twilight|Tour)\b',  # Product type
    ]

    has_address = bool(re.search(r'\d+\s+\w+', name_no_ext))
    has_location = bool(re.search(r'\b(CA|TX|FL|NY|AZ|CO|WA|OR|NV|California|Texas)\b', name_no_ext, re.I))

    if not has_address:
        issues.append("Missing property address in filename")

    # 6. Check if it looks like a temp file or export default name
    bad_patterns = [
        r'^IMG_\d+',
        r'^DSC_\d+',
        r'^MVI_\d+',
        r'^export',
        r'^final',
        r'^untitled',
        r'^\d{8}_\d{6}',  # Timestamp only
        r'^sequence',
    ]
    for pattern in bad_patterns:
        if re.match(pattern, name_no_ext, re.I):
            issues.append("Filename appears to be a default/temp name - needs proper naming")
            break

    if not issues:
        return QAIssue(
            check_name="Filename Convention",
            status="pass",
            message="Filename follows proper naming convention"
        )
    elif "Missing property address" in str(issues) or "default/temp name" in str(issues):
        return QAIssue(
            check_name="Filename Convention",
            status="fail",
            message="; ".join(issues),
            action="Rename file to: 'Address, City, State - Client Name - Product Type'"
        )
    else:
        return QAIssue(
            check_name="Filename Convention",
            status="warning",
            message="; ".join(issues),
            action="Review filename format"
        )


def check_lower_thirds(file_path: str, duration: float) -> QAIssue:
    """
    Check if the video contains lower thirds (text overlays with agent/property info).
    Analyzes frames for text-like elements in the lower portion of the screen.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return QAIssue(check_name="Lower Thirds", status="warning", message="OpenCV not available")

    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return QAIssue(check_name="Lower Thirds", status="warning", message="Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames in the first 15 seconds (lower thirds usually appear early)
        # and last 15 seconds (often appear at the end too)
        check_duration = min(15, duration * 0.3)
        frames_to_check = []

        # First section
        for t in np.arange(2, check_duration, 1):
            frames_to_check.append(int(t * fps))

        # Last section
        for t in np.arange(duration - check_duration, duration - 1, 1):
            frames_to_check.append(int(t * fps))

        lower_third_detected = False
        detection_times = []

        for frame_num in frames_to_check:
            if frame_num >= total_frames or frame_num < 0:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]

            # Focus on lower third of the frame
            lower_region = frame[int(h * 0.65):, :]

            # Convert to grayscale
            gray = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)

            # Look for high contrast areas (text is usually high contrast)
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Look for horizontal clusters of edges (text lines)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Also look for white/light colored regions (text boxes/lower thirds)
            white_mask = cv2.inRange(gray, 200, 255)
            white_ratio = np.sum(white_mask > 0) / (gray.shape[0] * gray.shape[1])

            # Detect potential text regions using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            dilated = cv2.dilate(edges, kernel, iterations=2)

            # Find contours that could be text blocks
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            text_like_regions = 0
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = cw / ch if ch > 0 else 0
                # Text blocks are usually wider than tall
                if aspect_ratio > 3 and cw > w * 0.1 and ch > 10:
                    text_like_regions += 1

            # Heuristic: lower third likely present if we have text-like regions
            if text_like_regions >= 1 or (edge_density > 0.05 and white_ratio > 0.02):
                lower_third_detected = True
                detection_times.append(frame_num / fps)

        cap.release()

        if lower_third_detected:
            return QAIssue(
                check_name="Lower Thirds",
                status="pass",
                message=f"Lower thirds detected (found around {detection_times[0]:.1f}s)"
            )
        else:
            return QAIssue(
                check_name="Lower Thirds",
                status="warning",
                message="No lower thirds detected - verify agent/property info is displayed",
                action="Add lower thirds with agent name, brokerage, and property address"
            )

    except Exception as e:
        return QAIssue(check_name="Lower Thirds", status="warning", message=f"Could not analyze: {str(e)}")


def check_sound_design(file_path: str, duration: float) -> QAIssue:
    """
    Check if video has sound design (sound effects, foley, ambient sounds)
    beyond just music. Analyzes frequency content and transients.
    """
    try:
        import numpy as np

        # Extract audio
        audio_path = file_path + '_temp_audio2.wav'
        cmd = [
            'ffmpeg', '-y', '-i', file_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
            audio_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)

        # Read audio
        import wave
        with wave.open(audio_path, 'rb') as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_data = np.frombuffer(wav.readframes(n_frames), dtype=np.int16).astype(np.float32)

        os.unlink(audio_path)

        if len(audio_data) == 0:
            return QAIssue(check_name="Sound Design", status="warning", message="No audio to analyze")

        # Analyze frequency content in chunks
        chunk_size = int(sample_rate * 0.5)  # 0.5 second chunks
        high_freq_energy = []
        mid_freq_energy = []
        low_freq_energy = []

        for i in range(0, len(audio_data) - chunk_size, chunk_size):
            chunk = audio_data[i:i + chunk_size]

            # FFT analysis
            fft = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1/sample_rate)

            # Frequency bands
            low_mask = freqs < 300
            mid_mask = (freqs >= 300) & (freqs < 3000)
            high_mask = freqs >= 3000

            low_freq_energy.append(np.mean(fft[low_mask]) if np.any(low_mask) else 0)
            mid_freq_energy.append(np.mean(fft[mid_mask]) if np.any(mid_mask) else 0)
            high_freq_energy.append(np.mean(fft[high_mask]) if np.any(high_mask) else 0)

        # Sound design typically adds:
        # - High frequency content (whooshes, impacts, textures)
        # - Varied transients (not just steady music)
        # - Distinct sound events at different times

        if not high_freq_energy:
            return QAIssue(check_name="Sound Design", status="warning", message="Could not analyze")

        avg_high = np.mean(high_freq_energy)
        high_variance = np.std(high_freq_energy)

        # Check for high frequency presence (sound effects tend to have more high freq content)
        # and variance (sound design adds variety, not just steady music)
        has_high_freq = avg_high > 100  # Threshold for high frequency presence
        has_variety = high_variance > avg_high * 0.3  # 30% variance indicates sound design

        # Detect sudden transients (impacts, whooshes)
        energy = [l + m + h for l, m, h in zip(low_freq_energy, mid_freq_energy, high_freq_energy)]
        energy_diff = np.abs(np.diff(energy))
        transient_count = np.sum(energy_diff > np.mean(energy_diff) * 2)

        if has_high_freq and has_variety and transient_count > 5:
            return QAIssue(
                check_name="Sound Design",
                status="pass",
                message=f"Rich sound design detected ({transient_count} sound events)"
            )
        elif has_high_freq or transient_count > 2:
            return QAIssue(
                check_name="Sound Design",
                status="warning",
                message="Basic sound design - could be enhanced",
                action="Consider adding whooshes, impacts, or ambient textures"
            )
        else:
            return QAIssue(
                check_name="Sound Design",
                status="info",
                message="Music only - no additional sound design detected",
                action="Add sound effects (whooshes, transitions, ambient) for richer audio"
            )

    except Exception as e:
        return QAIssue(check_name="Sound Design", status="warning", message=f"Could not analyze: {str(e)}")

def calculate_flow_rating(issues: list) -> QAIssue:
    """
    Calculate overall flow/vibe rating based on all checks.
    Combines stabilization, transitions, beat sync, and sound design scores.
    """
    # Weight each factor
    weights = {
        'Stabilization': 25,
        'Transitions': 20,
        'Beat Sync': 25,
        'Sound Design': 15,
        'Audio Levels': 15
    }

    score = 0
    max_score = 0
    factors = []

    for issue in issues:
        if issue.check_name in weights:
            weight = weights[issue.check_name]
            max_score += weight

            if issue.status == 'pass':
                score += weight
                factors.append(f"✓ {issue.check_name}")
            elif issue.status == 'warning':
                score += weight * 0.5
                factors.append(f"~ {issue.check_name}")
            else:
                factors.append(f"✗ {issue.check_name}")

    if max_score == 0:
        return QAIssue(check_name="Flow Rating", status="info", message="Not enough data to rate")

    percentage = (score / max_score) * 100

    # Convert to letter grade
    if percentage >= 90:
        grade = "A"
        status = "pass"
        vibe = "Excellent flow - this video vibes!"
    elif percentage >= 80:
        grade = "B+"
        status = "pass"
        vibe = "Great flow - minor improvements possible"
    elif percentage >= 70:
        grade = "B"
        status = "warning"
        vibe = "Good flow - some areas need work"
    elif percentage >= 60:
        grade = "C"
        status = "warning"
        vibe = "Decent flow - noticeable issues"
    else:
        grade = "D"
        status = "fail"
        vibe = "Flow needs significant improvement"

    return QAIssue(
        check_name="Flow Rating",
        status=status,
        message=f"{grade} ({percentage:.0f}%) - {vibe}",
        action="\n".join(factors) if percentage < 80 else None
    )

def check_video_filename(filename: str, folder_path: str = "") -> QAIssue:
    """Validate video filename follows convention"""
    # Expected pattern: Address, City, State - Agent Name - Brokerage (Type).mp4
    # Or for unbranded: Address, City, State - Type.mp4

    branded_pattern = r'^(.+),\s*(.+),\s*([A-Z]{2})\s*-\s*(.+)\s*-\s*(.+)\s*\((.+)\)\.mp4$'
    unbranded_pattern = r'^(.+),\s*(.+),\s*([A-Z]{2})\s*-\s*\((.+)\)\.mp4$'

    branded_match = re.match(branded_pattern, filename, re.IGNORECASE)
    unbranded_match = re.match(unbranded_pattern, filename, re.IGNORECASE)

    is_branded = 'branded' in folder_path.lower() or '1 -' in folder_path

    if branded_match:
        address, city, state, agent, brokerage, video_type = branded_match.groups()
        return QAIssue(
            check_name="File Naming",
            status="pass",
            message=f"Valid branded format",
            found=f"Address: {address}, {city}, {state} | Agent: {agent} | Brokerage: {brokerage} | Type: {video_type}"
        )
    elif unbranded_match and not is_branded:
        address, city, state, video_type = unbranded_match.groups()
        return QAIssue(
            check_name="File Naming",
            status="pass",
            message=f"Valid unbranded format",
            found=f"Address: {address}, {city}, {state} | Type: {video_type}"
        )
    else:
        return QAIssue(
            check_name="File Naming",
            status="fail",
            message="Filename does not match expected convention",
            expected="[Address], [City], [State] - [Agent] - [Brokerage] (Type).mp4",
            found=filename,
            action="Rename file to match convention"
        )

def parse_filename_components(filename: str) -> dict:
    """Extract components from filename for text verification"""
    components = {
        'address': None,
        'city': None,
        'state': None,
        'agent': None,
        'brokerage': None,
        'video_type': None
    }

    # Try branded pattern first
    branded_pattern = r'^(.+),\s*(.+),\s*([A-Z]{2})\s*-\s*(.+)\s*-\s*(.+)\s*\((.+)\)\.mp4$'
    match = re.match(branded_pattern, filename, re.IGNORECASE)

    if match:
        components['address'] = match.group(1).strip()
        components['city'] = match.group(2).strip()
        components['state'] = match.group(3).strip()
        components['agent'] = match.group(4).strip()
        components['brokerage'] = match.group(5).strip()
        components['video_type'] = match.group(6).strip()

    return components

# ============================================================================
# PHOTO ANALYSIS FUNCTIONS
# ============================================================================

def analyze_photo(file_path: str) -> List[QAIssue]:
    """
    Analyze a photo for QA issues based on Aerial Canvas Photo QA SOPs.

    Checks include:
    - File size (Print: 20-25MB, Web: ~2MB)
    - Resolution
    - Blur/Sharpness
    - Grain/Noise
    - Exposure (dark, overexposed, blown highlights)
    - Contrast & Vibrance
    - Color saturation (over/under)
    - White balance / Color casts
    - Vertical/Horizontal alignment
    - File naming convention
    """
    issues = []

    try:
        import cv2
        import numpy as np
        from PIL import Image

        # Load image
        img = cv2.imread(file_path)
        pil_img = Image.open(file_path)
        filename = os.path.basename(file_path)

        if img is None:
            return [QAIssue(check_name="Photo Load", status="fail", message="Could not load image")]

        height, width = img.shape[:2]

        # ===========================================
        # FILE SIZE CHECK (AC SOP: Print 20-25MB, Web 2MB)
        # ===========================================
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        is_print = 'print' in filename.lower()
        is_web = 'web' in filename.lower()

        if is_print:
            if 20 <= file_size_mb <= 25:
                issues.append(QAIssue(check_name="File Size", status="pass", message=f"{file_size_mb:.1f} MB (Print)"))
            else:
                issues.append(QAIssue(
                    check_name="File Size",
                    status="warning" if 15 <= file_size_mb <= 30 else "fail",
                    message=f"{file_size_mb:.1f} MB",
                    expected="20-25 MB for Print",
                    action="Adjust export settings"
                ))
        elif is_web:
            if 1.5 <= file_size_mb <= 2.5:
                issues.append(QAIssue(check_name="File Size", status="pass", message=f"{file_size_mb:.1f} MB (Web)"))
            else:
                issues.append(QAIssue(
                    check_name="File Size",
                    status="warning",
                    message=f"{file_size_mb:.1f} MB",
                    expected="~2 MB for Web",
                    action="Adjust compression"
                ))
        else:
            issues.append(QAIssue(check_name="File Size", status="info", message=f"{file_size_mb:.1f} MB"))

        # ===========================================
        # RESOLUTION CHECK
        # ===========================================
        long_edge = max(width, height)
        if long_edge >= 4000:
            issues.append(QAIssue(check_name="Resolution", status="pass", message=f"{width}x{height}"))
        elif long_edge >= 2000:
            issues.append(QAIssue(
                check_name="Resolution",
                status="warning",
                message=f"{width}x{height} - Lower than ideal",
                expected="4000px+ on long edge",
                found=f"{long_edge}px"
            ))
        else:
            issues.append(QAIssue(
                check_name="Resolution",
                status="fail",
                message=f"Resolution too low: {width}x{height}",
                expected="4000px+ on long edge",
                found=f"{long_edge}px",
                action="Use higher resolution source"
            ))

        # Convert to grayscale and HSV for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ===========================================
        # BLUR DETECTION (AC SOP: Blurry photos fail)
        # Uses regional analysis to distinguish:
        # - Full-frame blur (fail): everything soft
        # - Selective focus (warning): some sharp, some soft - may be intentional detail shot
        # - Sharp (pass): most regions sharp
        # ===========================================
        sharpness = analyze_regional_sharpness(gray)
        classification = sharpness['classification']
        mean_score = sharpness['mean_score']
        sharp_regions = sharpness['sharp_regions']
        soft_regions = sharpness['soft_regions']

        if classification == "sharp":
            issues.append(QAIssue(
                check_name="Sharpness",
                status="pass",
                message=f"Sharp ({sharp_regions}/9 regions crisp, score: {mean_score:.0f})"
            ))
        elif classification == "selective_focus":
            # Selective focus - might be intentional for detail shots
            issues.append(QAIssue(
                check_name="Sharpness",
                status="warning",
                message=f"Selective focus detected ({sharp_regions} sharp, {soft_regions} soft regions)",
                action="Review: OK for detail shots, reshoot if full room should be sharp"
            ))
        elif classification == "full_blur":
            # Full-frame blur - definitely a problem
            issues.append(QAIssue(
                check_name="Sharpness",
                status="fail",
                message=f"FULL-FRAME BLUR - entire image is soft (score: {mean_score:.0f})",
                action="Reshoot required - image is out of focus"
            ))
        elif classification == "soft":
            issues.append(QAIssue(
                check_name="Sharpness",
                status="warning",
                message=f"Image is soft (score: {mean_score:.0f})",
                action="Check focus - may need reshoot or sharpening"
            ))
        else:
            issues.append(QAIssue(
                check_name="Sharpness",
                status="pass",
                message=f"Acceptable sharpness (score: {mean_score:.0f})"
            ))

        # ===========================================
        # NOISE/GRAIN DETECTION (AC SOP: Grainy photos fail)
        # ===========================================
        noise_level = estimate_noise(gray)
        if noise_level < 5:
            issues.append(QAIssue(check_name="Noise/Grain", status="pass", message=f"Clean (score: {noise_level:.1f})"))
        elif noise_level < 10:
            issues.append(QAIssue(
                check_name="Noise/Grain",
                status="warning",
                message=f"Moderate grain (score: {noise_level:.1f})",
                action="Apply noise reduction in Lightroom/Photoshop"
            ))
        else:
            issues.append(QAIssue(
                check_name="Noise/Grain",
                status="fail",
                message=f"HIGH GRAIN detected (score: {noise_level:.1f})",
                action="Apply heavy noise reduction or reshoot with lower ISO"
            ))

        # ===========================================
        # HDR BLENDING ARTIFACT DETECTION
        # Detects: Halos, color banding, ghosting
        # ===========================================
        hdr_result = detect_hdr_artifacts(img)
        if hdr_result['has_artifacts']:
            artifact_messages = [a['message'] for a in hdr_result['artifacts']]
            artifact_types = [a['type'] for a in hdr_result['artifacts']]

            if hdr_result['severity'] == 'high':
                issues.append(QAIssue(
                    check_name="HDR Blending",
                    status="fail",
                    message=f"HDR artifacts detected: {', '.join(artifact_types)}",
                    action="Re-blend HDR or adjust processing. Issues: " + "; ".join(artifact_messages)
                ))
            else:
                issues.append(QAIssue(
                    check_name="HDR Blending",
                    status="warning",
                    message=f"Minor HDR artifacts: {', '.join(artifact_types)}",
                    action="Review blending quality. " + "; ".join(artifact_messages)
                ))
        else:
            issues.append(QAIssue(
                check_name="HDR Blending",
                status="pass",
                message="Clean HDR blend - no visible artifacts"
            ))

        # ===========================================
        # EXPORT ARTIFACT DETECTION
        # Macroblocking, color glitches, banding, corruption
        # ===========================================
        artifact_result = detect_export_artifacts(img)
        if artifact_result['has_artifacts']:
            artifact_types = [a['type'] for a in artifact_result['artifacts']]
            artifact_messages = [a['message'] for a in artifact_result['artifacts']]

            # Create preview with problem regions highlighted
            preview_img = None
            if artifact_result['problem_regions']:
                preview = img.copy()
                for region in artifact_result['problem_regions']:
                    rx, ry, rw, rh, label = region
                    cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (68, 68, 239), 3)  # Red in BGR
                    cv2.putText(preview, label, (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (68, 68, 239), 2)
                preview_img = create_photo_preview(preview, 0.5)

            if artifact_result['severity'] == 'high':
                issues.append(QAIssue(
                    check_name="Export Quality",
                    status="fail",
                    message=f"Export artifacts detected: {', '.join(set(artifact_types))}",
                    action="Re-export with higher quality settings. " + "; ".join(artifact_messages),
                    preview_image=preview_img
                ))
            else:
                issues.append(QAIssue(
                    check_name="Export Quality",
                    status="warning",
                    message=f"Minor export artifacts: {', '.join(set(artifact_types))}",
                    action="Consider re-exporting. " + "; ".join(artifact_messages),
                    preview_image=preview_img
                ))
        else:
            issues.append(QAIssue(
                check_name="Export Quality",
                status="pass",
                message="No compression or export artifacts detected"
            ))

        # ===========================================
        # EXPOSURE CHECK (AC SOP: Dark photos, Overexposed photos fail)
        # ===========================================
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        mean_brightness = np.mean(gray)

        # Check for clipped highlights (>250)
        highlight_clip = hist[250:].sum()
        # Check for crushed shadows (<5)
        shadow_clip = hist[:5].sum()

        # Check overall brightness
        exposure_issues = []

        if mean_brightness < 60:
            exposure_issues.append(f"Too DARK (brightness: {mean_brightness:.0f})")
        elif mean_brightness > 200:
            exposure_issues.append(f"OVEREXPOSED (brightness: {mean_brightness:.0f})")

        if highlight_clip > 0.05:
            exposure_issues.append(f"Blown highlights ({highlight_clip*100:.1f}%)")
        if shadow_clip > 0.05:
            exposure_issues.append(f"Crushed shadows ({shadow_clip*100:.1f}%)")

        if not exposure_issues:
            issues.append(QAIssue(check_name="Exposure", status="pass", message=f"Good exposure (brightness: {mean_brightness:.0f})"))
        else:
            is_severe = mean_brightness < 50 or mean_brightness > 210 or highlight_clip > 0.1 or shadow_clip > 0.1
            issues.append(QAIssue(
                check_name="Exposure",
                status="fail" if is_severe else "warning",
                message=", ".join(exposure_issues),
                action="Adjust exposure/recovery in Lightroom"
            ))

        # ===========================================
        # CONTRAST & VIBRANCE CHECK (AC SOP: Lack of contrast fails)
        # Note: Real estate often has bright whites - adjusted thresholds
        # ===========================================
        contrast = np.std(gray)

        if contrast > 40:
            issues.append(QAIssue(check_name="Contrast", status="pass", message=f"Good contrast (score: {contrast:.0f})"))
        elif contrast > 25:
            issues.append(QAIssue(
                check_name="Contrast",
                status="warning",
                message=f"Low contrast (score: {contrast:.0f})",
                action="Consider adding contrast if image looks flat"
            ))
        else:
            issues.append(QAIssue(
                check_name="Contrast",
                status="fail",
                message=f"Very low contrast (score: {contrast:.0f})",
                action="Image may need contrast adjustment"
            ))

        # ===========================================
        # SATURATION CHECK (AC SOP: Over/under saturated fails)
        # ===========================================
        saturation = hsv[:, :, 1]
        mean_sat = np.mean(saturation)

        # Check for oversaturation in warm tones (orange/yellow - common issue per SOP)
        # Hue range for orange/yellow is roughly 10-40 in OpenCV HSV
        warm_mask = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 40)
        if np.any(warm_mask):
            warm_saturation = np.mean(saturation[warm_mask])
        else:
            warm_saturation = 0

        sat_issues = []
        if mean_sat > 180:
            sat_issues.append(f"OVERSATURATED overall ({mean_sat:.0f})")
        elif mean_sat < 30:
            sat_issues.append(f"DESATURATED/flat ({mean_sat:.0f})")

        if warm_saturation > 200:
            sat_issues.append(f"Oversaturated orange/yellow tones ({warm_saturation:.0f})")

        if not sat_issues:
            issues.append(QAIssue(check_name="Saturation", status="pass", message=f"Good saturation ({mean_sat:.0f})"))
        else:
            issues.append(QAIssue(
                check_name="Saturation",
                status="fail" if mean_sat > 200 or mean_sat < 20 else "warning",
                message=", ".join(sat_issues),
                action="Adjust saturation - avoid oversaturated floors and warm tones"
            ))

        # ===========================================
        # WHITE BALANCE / COLOR CAST CHECK
        # (AC SOP: Color casts green/blue/magenta fail)
        # ===========================================
        b, g, r = cv2.mean(img)[:3]

        color_cast = None
        if g > r + 15 and g > b + 15:
            color_cast = "GREEN cast detected"
        elif b > r + 25:
            color_cast = "BLUE/COOL cast detected"
        elif r > b + 30 and r > g + 10:
            color_cast = "WARM/ORANGE cast detected"
        elif r > 100 and b > 100 and abs(r - b) < 10 and g < r - 15:
            color_cast = "MAGENTA cast detected"

        if color_cast:
            issues.append(QAIssue(
                check_name="White Balance",
                status="warning",
                message=color_cast,
                found=f"R:{r:.0f} G:{g:.0f} B:{b:.0f}",
                action="Correct white balance in Lightroom"
            ))
        else:
            issues.append(QAIssue(check_name="White Balance", status="pass", message="Neutral white balance"))

        # ===========================================
        # FILE NAMING CHECK (AC SOP naming convention)
        # ===========================================
        valid_prefixes = ['Photo_', 'Drone_', 'Twilight_', 'TwilightDrone_', 'Lifestyle_',
                         'PickupPhoto_', 'ReshootPhoto_']
        valid_suffixes = ['Print-', 'Web-']

        has_valid_prefix = any(filename.startswith(p) for p in valid_prefixes)
        has_valid_suffix = any(s in filename for s in valid_suffixes)

        if has_valid_prefix and has_valid_suffix:
            issues.append(QAIssue(check_name="File Naming", status="pass", message="Valid AC naming convention"))
        elif has_valid_prefix or has_valid_suffix:
            issues.append(QAIssue(
                check_name="File Naming",
                status="warning",
                message="Partial naming convention match",
                expected="Photo_Print-1.jpg or Photo_Web-1.jpg format",
                found=filename
            ))
        else:
            issues.append(QAIssue(
                check_name="File Naming",
                status="info",
                message="Non-standard filename",
                expected="[Type]_[Print/Web]-[#].jpg",
                found=filename
            ))

    except ImportError:
        issues.append(QAIssue(check_name="Photo Analysis", status="warning", message="OpenCV/PIL not installed"))
    except Exception as e:
        issues.append(QAIssue(check_name="Photo Analysis", status="fail", message=f"Error: {str(e)}"))

    return issues

def estimate_noise(gray_image) -> float:
    """Estimate noise level in image using Laplacian method"""
    import cv2
    import numpy as np

    # Use median absolute deviation of Laplacian
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sigma = np.median(np.abs(laplacian)) / 0.6745
    return sigma


def detect_export_artifacts(image) -> dict:
    """
    Detect common export/compression artifacts:
    - Macroblocking: Blocky 8x8 or 16x16 patterns from heavy compression
    - Color glitches: Random green/pink/corrupt color patches
    - Banding: Visible steps in gradients
    - Frame corruption: Partial damage or encoding errors

    Returns dict with artifact info and optional preview coordinates
    """
    import cv2
    import numpy as np

    artifacts = []
    severity = "none"
    problem_regions = []

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # ===========================================
    # 1. MACROBLOCKING DETECTION
    # Heavy JPEG/H.264 compression creates visible 8x8 or 16x16 blocks
    # ===========================================

    # Detect block boundaries using edge detection
    # Macroblocking shows up as a grid pattern of edges

    # Look for regular vertical and horizontal edge patterns
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # Sample along 8-pixel and 16-pixel boundaries
    block_score_8 = 0
    block_score_16 = 0

    # Check horizontal lines at 8px intervals
    for y in range(8, h - 8, 8):
        line_energy = np.mean(np.abs(sobel_h[y, :]))
        neighbors = (np.mean(np.abs(sobel_h[y-2, :])) + np.mean(np.abs(sobel_h[y+2, :]))) / 2
        if line_energy > neighbors * 1.5 and line_energy > 10:
            block_score_8 += 1

    # Check vertical lines at 8px intervals
    for x in range(8, w - 8, 8):
        line_energy = np.mean(np.abs(sobel_v[:, x]))
        neighbors = (np.mean(np.abs(sobel_v[:, x-2])) + np.mean(np.abs(sobel_v[:, x+2]))) / 2
        if line_energy > neighbors * 1.5 and line_energy > 10:
            block_score_8 += 1

    # Normalize by image size
    expected_lines = (h // 8) + (w // 8)
    block_ratio = block_score_8 / expected_lines if expected_lines > 0 else 0

    if block_ratio > 0.4:
        artifacts.append({
            'type': 'macroblocking',
            'severity': 'high' if block_ratio > 0.6 else 'medium',
            'message': f'Compression blocking artifacts detected ({block_ratio*100:.0f}% of block boundaries visible)'
        })
        severity = 'high' if block_ratio > 0.6 else 'medium'

    # ===========================================
    # 2. COLOR GLITCH DETECTION
    # Encoding errors often produce patches of wrong colors
    # (bright green, pink, or other out-of-place colors)
    # ===========================================

    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Look for patches of highly saturated green (encoding error signature)
        # Green glitches: H around 60, high S, high V
        green_mask = cv2.inRange(hsv, np.array([35, 200, 100]), np.array([85, 255, 255]))
        green_pixels = np.sum(green_mask > 0)
        green_ratio = green_pixels / (h * w)

        # Look for patches of bright pink/magenta (another common glitch)
        pink_mask = cv2.inRange(hsv, np.array([140, 150, 100]), np.array([170, 255, 255]))
        pink_pixels = np.sum(pink_mask > 0)
        pink_ratio = pink_pixels / (h * w)

        # Small patches of these colors in otherwise normal images = glitch
        if green_ratio > 0.001 and green_ratio < 0.1:
            # Find the glitch region
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Significant patch
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    problem_regions.append((x, y, cw, ch, "Green glitch"))
                    artifacts.append({
                        'type': 'color_glitch',
                        'severity': 'high',
                        'message': f'Green color glitch detected (encoding error)',
                        'coords': (x, y, cw, ch)
                    })
                    severity = 'high'
                    break

        if pink_ratio > 0.001 and pink_ratio < 0.1:
            contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    problem_regions.append((x, y, cw, ch, "Pink glitch"))
                    artifacts.append({
                        'type': 'color_glitch',
                        'severity': 'high',
                        'message': f'Pink/magenta color glitch detected (encoding error)',
                        'coords': (x, y, cw, ch)
                    })
                    severity = 'high'
                    break

    # ===========================================
    # 3. GRADIENT BANDING DETECTION
    # Visible steps in smooth gradients (bit depth or compression issue)
    # ===========================================

    # Look at sky region (top third) for banding
    sky_region = gray[:h//3, :]

    if sky_region.size > 0:
        # Calculate histogram of sky region
        hist = cv2.calcHist([sky_region], [0], None, [256], [0, 256]).flatten()

        # Banding shows up as spiky histogram with gaps
        # Smooth gradients have smooth histograms
        non_zero_bins = np.sum(hist > 0)
        total_bins = 256

        # Check for gaps in the histogram
        hist_smoothed = np.convolve(hist, np.ones(5)/5, mode='same')
        gaps = np.sum((hist == 0) & (hist_smoothed > 10))

        if gaps > 30 and non_zero_bins < 100:
            artifacts.append({
                'type': 'banding',
                'severity': 'medium',
                'message': f'Gradient banding detected in sky/smooth areas ({gaps} color gaps)'
            })
            if severity == 'none':
                severity = 'medium'

    # ===========================================
    # 4. PARTIAL FRAME CORRUPTION
    # Rows or columns of obviously wrong data
    # ===========================================

    # Check for rows that are drastically different from neighbors
    row_means = np.mean(gray, axis=1)
    row_diffs = np.abs(np.diff(row_means))

    corrupt_rows = np.where(row_diffs > 50)[0]
    if len(corrupt_rows) > 0 and len(corrupt_rows) < 10:
        # A few drastically different rows = corruption
        artifacts.append({
            'type': 'frame_corruption',
            'severity': 'high',
            'message': f'Possible frame corruption detected ({len(corrupt_rows)} abnormal rows)'
        })
        severity = 'high'
        # Mark first corrupt area
        if len(corrupt_rows) > 0:
            y = corrupt_rows[0]
            problem_regions.append((0, max(0, y-5), w, 10, "Corrupt"))

    return {
        'has_artifacts': len(artifacts) > 0,
        'severity': severity,
        'artifacts': artifacts,
        'problem_regions': problem_regions
    }


def analyze_regional_sharpness(gray_image) -> dict:
    """
    Analyze sharpness in multiple regions to distinguish:
    - Full-frame blur (bad): all regions are soft
    - Selective focus (review): some regions sharp, some soft (may be intentional)
    - Sharp image (good): most regions are sharp

    Returns dict with regional scores and classification
    """
    import cv2
    import numpy as np

    h, w = gray_image.shape

    # Divide image into 3x3 grid (9 regions)
    regions = []
    region_scores = []

    rows, cols = 3, 3
    rh, rw = h // rows, w // cols

    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * rh, (i + 1) * rh
            x1, x2 = j * rw, (j + 1) * rw
            region = gray_image[y1:y2, x1:x2]

            # Calculate Laplacian variance for this region
            lap_var = cv2.Laplacian(region, cv2.CV_64F).var()
            region_scores.append(lap_var)
            regions.append({'row': i, 'col': j, 'score': lap_var})

    # Calculate statistics
    scores = np.array(region_scores)
    mean_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    std_score = np.std(scores)

    # Count sharp vs soft regions
    sharp_threshold = 80  # Regions above this are considered sharp
    soft_threshold = 40   # Regions below this are considered soft

    sharp_count = np.sum(scores > sharp_threshold)
    soft_count = np.sum(scores < soft_threshold)

    # Classify the image
    if mean_score > 100 and sharp_count >= 7:
        classification = "sharp"
        confidence = "high"
    elif mean_score > 60 and sharp_count >= 5:
        classification = "sharp"
        confidence = "medium"
    elif max_score > 150 and soft_count >= 4 and sharp_count >= 2:
        # Some regions very sharp, some very soft = selective focus
        classification = "selective_focus"
        confidence = "high" if std_score > 60 else "medium"
    elif mean_score < 40 and max_score < 80:
        # All regions are soft = full-frame blur
        classification = "full_blur"
        confidence = "high"
    elif mean_score < 60:
        classification = "soft"
        confidence = "medium"
    else:
        classification = "acceptable"
        confidence = "medium"

    return {
        'classification': classification,
        'confidence': confidence,
        'mean_score': mean_score,
        'max_score': max_score,
        'min_score': min_score,
        'std_score': std_score,
        'sharp_regions': int(sharp_count),
        'soft_regions': int(soft_count),
        'regions': regions
    }


def detect_hdr_artifacts(image) -> dict:
    """
    Detect common HDR blending artifacts:
    - Halos: Bright/dark borders around high-contrast edges
    - Color banding: Abrupt color transitions in gradients (sky, walls)
    - Ghosting: Semi-transparent duplications from movement between frames

    Returns dict with artifact detection results
    """
    import cv2
    import numpy as np

    artifacts = []
    severity = "none"

    # Convert to different color spaces for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
    else:
        gray = image
        l_channel = gray

    h, w = gray.shape

    # ===========================================
    # 1. HALO DETECTION
    # Halos appear as bright/dark bands along edges
    # ===========================================

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Dilate edges to create a band around them
    kernel = np.ones((5, 5), np.uint8)
    edge_band = cv2.dilate(edges, kernel, iterations=2)
    edge_band_inner = cv2.dilate(edges, kernel, iterations=1)

    # Get the outer ring (potential halo zone)
    halo_zone = edge_band - edge_band_inner

    if np.sum(halo_zone > 0) > 100:  # Need enough edge pixels
        # Check for unusually bright or dark pixels in halo zone
        halo_pixels = l_channel[halo_zone > 0]
        non_halo_pixels = l_channel[edge_band == 0]

        if len(halo_pixels) > 0 and len(non_halo_pixels) > 0:
            halo_mean = np.mean(halo_pixels)
            non_halo_mean = np.mean(non_halo_pixels)

            # Halos often show as significantly brighter areas around edges
            brightness_diff = abs(halo_mean - non_halo_mean)

            # Check variance in halo zone (halos have unnatural uniformity)
            halo_std = np.std(halo_pixels)

            if brightness_diff > 30 and halo_std < 40:
                artifacts.append({
                    'type': 'halo',
                    'severity': 'high' if brightness_diff > 50 else 'medium',
                    'message': f'Possible HDR halos detected (edge brightness diff: {brightness_diff:.0f})'
                })
                severity = "high" if brightness_diff > 50 else "medium"

    # ===========================================
    # 2. COLOR BANDING DETECTION
    # Look for abrupt transitions in gradients
    # ===========================================

    # Focus on top third (usually sky in real estate photos)
    sky_region = l_channel[:h//3, :]

    if sky_region.size > 0:
        # Calculate horizontal gradient
        grad_x = np.abs(np.diff(sky_region.astype(float), axis=1))

        # Look for sudden jumps in otherwise smooth areas
        # Smooth gradient should have small consistent values
        # Banding shows as periodic spikes

        row_max_grads = np.max(grad_x, axis=1)
        row_mean_grads = np.mean(grad_x, axis=1)

        # Check for rows with suspiciously uniform gradients interrupted by jumps
        potential_banding_rows = np.sum((row_max_grads > 20) & (row_mean_grads < 5))

        banding_ratio = potential_banding_rows / len(row_max_grads) if len(row_max_grads) > 0 else 0

        if banding_ratio > 0.3:
            artifacts.append({
                'type': 'banding',
                'severity': 'high' if banding_ratio > 0.5 else 'medium',
                'message': f'Possible color banding in sky/gradient areas ({banding_ratio*100:.0f}% of rows affected)'
            })
            if severity != "high":
                severity = "high" if banding_ratio > 0.5 else "medium"

    # ===========================================
    # 3. GHOSTING DETECTION
    # Look for semi-transparent edges or doubled features
    # ===========================================

    # Ghosting often shows up as blurred edges with secondary faint edges nearby
    # Use edge detection at multiple thresholds

    edges_strong = cv2.Canny(gray, 100, 200)
    edges_weak = cv2.Canny(gray, 30, 80)

    # Ghost edges would appear in weak but not strong
    potential_ghosts = edges_weak - edges_strong
    potential_ghosts[potential_ghosts < 0] = 0

    # Calculate ratio of ghost edges to real edges
    strong_edge_count = np.sum(edges_strong > 0)
    ghost_edge_count = np.sum(potential_ghosts > 0)

    if strong_edge_count > 1000:  # Need enough edges to analyze
        ghost_ratio = ghost_edge_count / strong_edge_count

        if ghost_ratio > 3.0:
            artifacts.append({
                'type': 'ghosting',
                'severity': 'high' if ghost_ratio > 5.0 else 'medium',
                'message': f'Possible HDR ghosting detected (faint edge ratio: {ghost_ratio:.1f}x)'
            })
            if severity != "high":
                severity = "high" if ghost_ratio > 5.0 else "medium"

    return {
        'has_artifacts': len(artifacts) > 0,
        'severity': severity,
        'artifacts': artifacts,
        'artifact_count': len(artifacts)
    }


# ============================================================================
# MAIN QA FUNCTION
# ============================================================================

def run_video_qa(file_path: str, folder_path: str = "", progress_callback=None, original_filename: str = None, analysis_mode: str = "standard") -> QAReport:
    """Run QA analysis on a video file with configurable analysis depth.

    Args:
        file_path: Path to video file
        folder_path: Optional folder path for naming checks
        progress_callback: Callback for progress updates
        original_filename: Original filename (for Dropbox downloads)
        analysis_mode: "quick", "standard", or "full"
            - quick: ~30 sec - Basic checks only (technical, audio levels, black frames sampled)
            - standard: ~2-3 min - Most checks except motion analysis
            - full: ~5-10 min - All checks including motion/stabilization analysis
    """
    filename = original_filename if original_filename else os.path.basename(file_path)
    issues = []

    # Define which checks to run based on mode
    run_motion_checks = analysis_mode == "full"
    run_detailed_audio = analysis_mode in ("standard", "full")
    run_color_checks = analysis_mode in ("standard", "full")
    run_edit_checks = analysis_mode in ("standard", "full")
    sample_black_frames = analysis_mode == "quick"  # Sample instead of full scan

    # Calculate total steps based on mode
    if analysis_mode == "quick":
        total_steps = 10
    elif analysis_mode == "standard":
        total_steps = 16
    else:  # full
        total_steps = 20

    current_step = 0

    def update_progress(message):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            # Cap progress at 0.99 to prevent overflow, final step will set to 1.0
            progress = min(current_step / total_steps, 0.99)
            progress_callback(progress, message)

    # Step 1: Read metadata (ALL MODES)
    update_progress("Reading video metadata...")
    metadata = get_video_metadata(file_path)
    if 'error' in metadata:
        return QAReport(
            filename=filename,
            file_type="video",
            overall_status="fail",
            issues=[QAIssue(check_name="Metadata", status="fail", message=metadata['error'])],
            metadata={'analysis_mode': analysis_mode}
        )

    # Detect video format (aspect ratio, vertical vs horizontal, etc.)
    format_info = detect_video_format(metadata)
    is_vertical = format_info.get('is_vertical', False)
    is_cinematic = format_info.get('is_cinematic', False)
    duration = float(metadata.get('format', {}).get('duration', 0))

    # Step 2: Check aspect ratio and format type (ALL MODES)
    update_progress("Detecting video format...")
    issues.append(check_aspect_ratio(metadata, filename))

    # Step 3: Check resolution (ALL MODES)
    update_progress("Checking resolution...")
    issues.append(check_video_resolution(metadata, is_vertical, format_info))

    # Step 4: Check frame rate (ALL MODES)
    update_progress("Checking frame rate...")
    issues.append(check_frame_rate(metadata))

    # Step 5: Check audio presence (ALL MODES)
    update_progress("Checking audio...")
    issues.append(check_audio_present(metadata))

    # Step 6: Check audio levels (ALL MODES)
    update_progress("Analyzing audio levels...")
    issues.append(check_audio_levels(file_path))

    # Step 7: Detect black frames (ALL MODES - but sampled in quick mode)
    update_progress("Scanning for black frames...")
    issues.append(detect_black_frames(file_path, duration, sample_only=sample_black_frames))

    # Step 8: Detect log footage (STANDARD + FULL)
    if run_color_checks:
        update_progress("Analyzing color grading...")
        issues.append(detect_log_footage(file_path, duration))

    # Step 9: Check fade out (STANDARD + FULL)
    if run_edit_checks:
        update_progress("Checking fade out...")
        issues.append(check_fade_out(file_path, duration))

    # Step 10: Check stabilization - MOST TAXING (FULL ONLY)
    if run_motion_checks:
        update_progress("Analyzing motion stability...")
        issues.append(check_stabilization(file_path, duration))

    # Step 11: Check transition smoothness - TAXING (FULL ONLY)
    if run_motion_checks:
        update_progress("Analyzing transitions...")
        issues.append(check_transition_smoothness(file_path, duration))

    # Step 12: Check beat sync (STANDARD + FULL)
    if run_detailed_audio:
        update_progress("Checking beat synchronization...")
        issues.append(check_beat_sync(file_path, duration))

    # Step 13: Check sound design (STANDARD + FULL)
    if run_detailed_audio:
        update_progress("Analyzing sound design...")
        issues.append(check_sound_design(file_path, duration))

    # Step 14: Check audio fade in/out (STANDARD + FULL)
    if run_detailed_audio:
        update_progress("Checking music fades...")
        issues.append(check_audio_fade_in_out(file_path, duration))

    # Step 15: Check audio noise (STANDARD + FULL)
    if run_detailed_audio:
        update_progress("Analyzing audio quality...")
        issues.append(check_audio_noise(file_path))

    # Step 16: Check color consistency (STANDARD + FULL)
    if run_color_checks:
        update_progress("Checking color consistency...")
        issues.append(check_color_consistency(file_path, duration))

    # Step 17: Check for lower thirds (STANDARD + FULL)
    if run_edit_checks:
        update_progress("Detecting lower thirds...")
        issues.append(check_lower_thirds(file_path, duration))

    # Filename checks (ALL MODES)
    update_progress("Validating filename...")
    issues.append(check_video_filename(filename, folder_path))

    update_progress("Checking naming convention...")
    issues.append(check_filename_convention(filename, folder_path))

    # Calculate overall flow rating (ALL MODES)
    update_progress("Calculating results...")
    issues.append(calculate_flow_rating(issues))

    # Determine overall status
    has_failures = any(i.status == "fail" for i in issues)
    overall_status = "fail" if has_failures else "pass"

    # Track stats
    stats_tracker.increment_stat('total_videos_analyzed')
    issue_count = sum(1 for i in issues if i.status in ('fail', 'warning'))
    stats_tracker.increment_stat('total_issues_found', issue_count)

    # Mode labels for display
    mode_labels = {
        'quick': 'Quick Scan',
        'standard': 'Standard',
        'full': 'Full Analysis'
    }

    return QAReport(
        filename=filename,
        file_type="video",
        overall_status=overall_status,
        issues=issues,
        metadata={
            'duration': duration,
            'duration_formatted': f"{int(duration//60)}:{duration%60:05.2f}",
            'width': format_info.get('width', 0),
            'height': format_info.get('height', 0),
            'fps': format_info.get('fps', 0),
            'aspect_ratio': format_info.get('aspect_ratio_name', 'Unknown'),
            'format_type': format_info.get('format_type', 'horizontal'),
            'is_vertical': is_vertical,
            'is_cinematic': is_cinematic,
            'resolution_tier': format_info.get('resolution_tier', 'Unknown'),
            'use_case': format_info.get('use_case', 'Unknown'),
            'analysis_mode': analysis_mode,
            'analysis_mode_label': mode_labels.get(analysis_mode, 'Standard')
        }
    )

def run_photo_qa(file_path: str, progress_callback=None, skip_filename_check: bool = False, original_filename: str = None) -> QAReport:
    """Run complete QA analysis on a photo with progress updates

    Args:
        file_path: Path to the photo file
        progress_callback: Optional callback for progress updates
        skip_filename_check: Skip filename validation (useful for batch/Dropbox downloads)
        original_filename: Use this filename instead of temp file name (for Dropbox downloads)
    """
    filename = original_filename if original_filename else os.path.basename(file_path)

    def update_progress(step, total, message):
        if progress_callback:
            progress_callback(step / total, message)

    update_progress(1, 18, "Loading image...")

    try:
        import cv2
        import numpy as np
        from PIL import Image

        img = cv2.imread(file_path)
        if img is None:
            return QAReport(
                filename=filename,
                file_type="photo",
                overall_status="fail",
                issues=[QAIssue(check_name="Load", status="fail", message="Could not load image")],
                metadata={}
            )

        issues = []
        height, width = img.shape[:2]

        # Downscale for faster analysis on large images (keep original for resolution check)
        max_analysis_size = 1500  # Analyze at max 1500px for speed
        scale_factor = 1.0
        if max(height, width) > max_analysis_size:
            scale_factor = max_analysis_size / max(height, width)
            img_small = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        else:
            img_small = img

        # Use downscaled versions for analysis (much faster)
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

        # Step 2: File size
        update_progress(2, 18, "Checking file size...")
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        is_print = 'print' in filename.lower()
        is_web = 'web' in filename.lower()
        if is_print:
            if 20 <= file_size_mb <= 25:
                issues.append(QAIssue(check_name="File Size", status="pass", message=f"{file_size_mb:.1f} MB (Print)"))
            else:
                issues.append(QAIssue(check_name="File Size", status="warning" if 15 <= file_size_mb <= 30 else "fail",
                    message=f"{file_size_mb:.1f} MB", expected="20-25 MB for Print", action="Adjust export settings"))
        elif is_web:
            if 1.5 <= file_size_mb <= 2.5:
                issues.append(QAIssue(check_name="File Size", status="pass", message=f"{file_size_mb:.1f} MB (Web)"))
            else:
                issues.append(QAIssue(check_name="File Size", status="warning", message=f"{file_size_mb:.1f} MB",
                    expected="~2 MB for Web", action="Adjust compression"))
        else:
            issues.append(QAIssue(check_name="File Size", status="info", message=f"{file_size_mb:.1f} MB"))

        # Step 3: Resolution
        update_progress(3, 18, "Checking resolution...")
        long_edge = max(width, height)
        if long_edge >= 4000:
            issues.append(QAIssue(check_name="Resolution", status="pass", message=f"{width}x{height}"))
        elif long_edge >= 2000:
            issues.append(QAIssue(check_name="Resolution", status="warning", message=f"{width}x{height} - Lower than ideal",
                expected="4000px+ on long edge", found=f"{long_edge}px"))
        else:
            issues.append(QAIssue(check_name="Resolution", status="fail", message=f"Resolution too low: {width}x{height}",
                expected="4000px+ on long edge", found=f"{long_edge}px", action="Use higher resolution source"))

        # Step 4: Sharpness (adjusted for real estate - large uniform areas like walls lower the score)
        update_progress(4, 18, "Analyzing sharpness...")
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 50:
            issues.append(QAIssue(check_name="Sharpness", status="pass", message=f"Sharp (score: {laplacian_var:.0f})"))
        elif laplacian_var > 25:
            issues.append(QAIssue(check_name="Sharpness", status="warning", message=f"Slightly soft (score: {laplacian_var:.0f})",
                action="Check focus - may need sharpening in post"))
        else:
            issues.append(QAIssue(check_name="Sharpness", status="fail", message=f"Image appears BLURRY (score: {laplacian_var:.0f})",
                action="Reshoot required - image is too blurry"))

        # Step 5: Noise (adjusted - normal camera grain is acceptable)
        update_progress(5, 18, "Analyzing noise levels...")
        noise_level = estimate_noise(gray)
        if noise_level < 12:
            issues.append(QAIssue(check_name="Noise/Grain", status="pass", message=f"Clean (score: {noise_level:.1f})"))
        elif noise_level < 20:
            issues.append(QAIssue(check_name="Noise/Grain", status="warning", message=f"Moderate grain (score: {noise_level:.1f})",
                action="Consider noise reduction if visible"))
        else:
            issues.append(QAIssue(check_name="Noise/Grain", status="fail", message=f"HIGH GRAIN detected (score: {noise_level:.1f})",
                action="Apply noise reduction or reshoot with lower ISO"))

        # Step 6: Exposure
        update_progress(6, 18, "Checking exposure...")
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten() / gray.size
        mean_brightness = np.mean(gray)
        highlight_clip = hist[250:].sum()
        shadow_clip = hist[:5].sum()
        exposure_issues = []
        if mean_brightness < 60:
            exposure_issues.append(f"Too DARK (brightness: {mean_brightness:.0f})")
        elif mean_brightness > 200:
            exposure_issues.append(f"OVEREXPOSED (brightness: {mean_brightness:.0f})")
        if highlight_clip > 0.05:
            exposure_issues.append(f"Blown highlights ({highlight_clip*100:.1f}%)")
        if shadow_clip > 0.05:
            exposure_issues.append(f"Crushed shadows ({shadow_clip*100:.1f}%)")
        if not exposure_issues:
            issues.append(QAIssue(check_name="Exposure", status="pass", message=f"Good exposure (brightness: {mean_brightness:.0f})"))
        else:
            is_severe = mean_brightness < 50 or mean_brightness > 210 or highlight_clip > 0.1 or shadow_clip > 0.1
            issues.append(QAIssue(check_name="Exposure", status="fail" if is_severe else "warning",
                message=", ".join(exposure_issues), action="Adjust exposure/recovery in Lightroom"))

        # Step 7: Contrast
        # Note: Real estate photos often have bright white walls which naturally lower contrast scores
        # Thresholds adjusted to be forgiving for intentionally bright/high-key photos
        update_progress(7, 18, "Checking contrast...")
        contrast = np.std(gray)
        if contrast > 40:
            issues.append(QAIssue(check_name="Contrast", status="pass", message=f"Good contrast (score: {contrast:.0f})"))
        elif contrast > 25:
            issues.append(QAIssue(check_name="Contrast", status="warning", message=f"Low contrast (score: {contrast:.0f})",
                action="Consider adding contrast if image looks flat"))
        else:
            issues.append(QAIssue(check_name="Contrast", status="fail", message=f"Very low contrast (score: {contrast:.0f})",
                action="Image may need contrast adjustment"))

        # Step 8: Saturation
        update_progress(8, 18, "Checking saturation...")
        saturation = hsv[:, :, 1]
        mean_sat = np.mean(saturation)
        if mean_sat > 180:
            issues.append(QAIssue(check_name="Saturation", status="fail", message=f"OVERSATURATED ({mean_sat:.0f})",
                action="Reduce saturation"))
        elif mean_sat < 30:
            issues.append(QAIssue(check_name="Saturation", status="warning", message=f"Low saturation ({mean_sat:.0f})",
                action="May need color boost"))
        else:
            issues.append(QAIssue(check_name="Saturation", status="pass", message=f"Good saturation ({mean_sat:.0f})"))

        # Step 9: White balance
        update_progress(9, 18, "Checking white balance...")
        b, g, r = cv2.mean(img)[:3]
        color_cast = None
        if g > r + 15 and g > b + 15:
            color_cast = "GREEN cast"
        elif b > r + 25:
            color_cast = "BLUE/COOL cast"
        elif r > b + 30 and r > g + 10:
            color_cast = "WARM/ORANGE cast"
        if color_cast:
            issues.append(QAIssue(check_name="White Balance", status="warning", message=color_cast,
                action="Correct white balance"))
        else:
            issues.append(QAIssue(check_name="White Balance", status="pass", message="Neutral"))

        # Step 10: AI Over-Processing / Grayscale Detection
        # Detects when AI editing removes too much color, making image look nearly grayscale
        update_progress(10, 18, "Checking for AI over-processing...")
        # Calculate color variance - how much color variation exists
        sat_std = np.std(saturation)
        # Check if image is nearly grayscale (very low saturation with low variance)
        # Normal photos: mean_sat ~50-120, sat_std ~30-60
        # Over-processed/grayscale: mean_sat <25, sat_std <15
        is_nearly_grayscale = mean_sat < 25 and sat_std < 20
        # Also check for flat/washed-out look (low contrast + low saturation)
        is_washed_out = mean_sat < 35 and contrast < 40

        if is_nearly_grayscale:
            issues.append(QAIssue(
                check_name="AI Processing",
                status="fail",
                message=f"Image appears GRAYSCALE/desaturated (sat: {mean_sat:.0f}, var: {sat_std:.0f})",
                action="AI editing removed too much color - restore saturation or re-edit"
            ))
        elif is_washed_out:
            issues.append(QAIssue(
                check_name="AI Processing",
                status="warning",
                message=f"Image looks washed out (sat: {mean_sat:.0f}, contrast: {contrast:.0f})",
                action="May need color/contrast boost - check if AI over-edited"
            ))
        else:
            issues.append(QAIssue(check_name="AI Processing", status="pass", message="Colors look natural"))

        # Step 11: Reflection Detection (cameras, people, equipment in mirrors/windows)
        update_progress(11, 18, "Scanning for reflections...")
        reflection_detected = False
        reflection_details = []

        # Simple approach: detect high-contrast circular/rectangular patterns that could be cameras
        # and skin-tone colored regions in areas that look like reflective surfaces

        # Detect potential mirror/window regions (high luminance variance, rectangular)
        # Look for skin tones in the image (potential photographer reflection)
        # HSV ranges for skin tones
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_percentage = (np.sum(skin_mask > 0) / skin_mask.size) * 100

        # If significant skin tones detected, could indicate people in frame
        # Real estate photos shouldn't have people visible (including reflections)
        if skin_percentage > 2.0:
            reflection_details.append(f"Possible person/skin tones detected ({skin_percentage:.1f}% of image)")
            reflection_detected = True

        # Look for camera-like dark circular objects with bright spots (lens reflections)
        # This is a simplified heuristic - real detection would need ML
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 50,
                                   param1=50, param2=30, minRadius=10, maxRadius=100)
        if circles is not None and len(circles[0]) > 3:
            # Multiple circular objects could indicate camera equipment
            reflection_details.append(f"Multiple circular objects detected ({len(circles[0])} found)")

        if reflection_detected:
            issues.append(QAIssue(
                check_name="Reflections",
                status="warning",
                message="; ".join(reflection_details),
                action="Check mirrors/windows for photographer, camera, or tripod reflections"
            ))
        else:
            issues.append(QAIssue(check_name="Reflections", status="pass", message="No obvious reflections detected"))

        # Step 12: Bathroom Staging Check
        # Detect toilet paper rolls, toilet seats up, and other staging issues
        update_progress(12, 18, "Checking for staging issues...")
        staging_issues = []
        staging_boxes = []  # Store (x, y, w, h, label) for each detected issue

        # Use downscaled image for faster staging detection
        hsv_staging = hsv  # Already using downscaled hsv

        # Detect white objects (potential toilet paper, toilets)
        # White has low saturation and high value
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv_staging, lower_white, upper_white)

        # Find contours of white objects
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for circular/cylindrical white objects (toilet paper rolls)
        # They're typically small-medium sized and roughly circular
        # Scale area thresholds based on image size
        area_scale = (scale_factor ** 2) if scale_factor < 1.0 else 1.0
        min_area = int(500 * area_scale)
        max_area = int(50000 * area_scale)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:  # Filter by reasonable size (scaled)
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Get bounding box aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Toilet paper rolls are roughly circular or slightly oval
            # and typically in the middle-right or corner areas
            if circularity > 0.6 and 0.7 < aspect_ratio < 1.4:
                # Check if it's in a location consistent with bathroom (not center of image)
                img_h, img_w = img_small.shape[:2]
                center_x = x + w/2
                center_y = y + h/2

                # Toilet paper usually not in center of frame
                if not (0.3 < center_x/img_w < 0.7 and 0.3 < center_y/img_h < 0.7):
                    # Could be toilet paper - flag for review (scaled thresholds)
                    tp_min = int(1000 * area_scale)
                    tp_max = int(15000 * area_scale)
                    if tp_min < area < tp_max:
                        staging_issues.append("Possible toilet paper roll visible")
                        staging_boxes.append((x, y, w, h, "TP"))
                        break

        # Look for toilet seat up - characterized by a U-shape or open oval
        # Toilet seats up create a distinctive white U or horseshoe shape
        toilet_min = int(5000 * area_scale)
        toilet_max = int(100000 * area_scale)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < toilet_min or area > toilet_max:  # Toilet-sized (scaled)
                continue

            # Check if shape is roughly U-shaped (open at top or bottom)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Toilet seats are typically wider than tall when viewed from front
            if 0.8 < aspect_ratio < 2.0:
                # Check the fill ratio (U-shape has lower fill than solid oval)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    # U-shape has solidity around 0.5-0.8 (not fully filled)
                    if 0.4 < solidity < 0.85:
                        # Check if it's white and has a gap (the open part of U)
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        mean_val = cv2.mean(gray, mask=mask)[0]
                        if mean_val > 180:  # Very white
                            staging_issues.append("Possible toilet seat up")
                            staging_boxes.append((x, y, w, h, "Toilet"))
                            break

        # Check for other staging issues - trash cans visible, clutter
        # Look for dark rectangular objects near edges (trash cans)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 60])
        dark_mask = cv2.inRange(hsv_staging, lower_dark, upper_dark)
        dark_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in dark_contours:
            area = cv2.contourArea(contour)
            if area < 3000 or area > 80000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(h) / w if w > 0 else 0

            # Trash cans are typically taller than wide
            if aspect_ratio > 1.3 and aspect_ratio < 3.0:
                img_h, img_w = img_small.shape[:2]
                # Usually near edges/corners
                if x < img_w * 0.15 or x + w > img_w * 0.85:
                    staging_issues.append("Possible trash can or dark object near edge")
                    staging_boxes.append((x, y, w, h, "Trash"))
                    break

        if staging_issues:
            # Remove duplicates
            staging_issues = list(set(staging_issues))

            # Create preview image with bounding boxes
            preview_img = None
            if staging_boxes:
                preview = img.copy()
                for box in staging_boxes:
                    bx, by, bw, bh, label = box
                    # Draw rectangle in warning yellow
                    cv2.rectangle(preview, (bx, by), (bx + bw, by + bh), (11, 158, 245), 3)
                    # Add label
                    cv2.putText(preview, label, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (11, 158, 245), 2)
                preview_img = create_photo_preview(preview, 0.5)

            issues.append(QAIssue(
                check_name="Staging",
                status="warning",
                message="; ".join(staging_issues),
                action="Review image for bathroom staging issues - toilet paper, seat up, trash cans",
                preview_image=preview_img,
                preview_coords=staging_boxes[0][:4] if staging_boxes else None
            ))
        else:
            issues.append(QAIssue(check_name="Staging", status="pass", message="No obvious staging issues"))

        # Step 13: Window View Quality - Check if windows are blown out
        update_progress(13, 18, "Checking window exposures...")
        try:
            # Detect potential window regions (bright rectangular areas)
            # Windows typically appear as bright rectangular regions
            bright_threshold = 240
            bright_mask = (gray > bright_threshold).astype(np.uint8) * 255
            window_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            blown_windows = 0
            total_bright_area = 0
            img_area = gray.shape[0] * gray.shape[1]

            for contour in window_contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # Skip small bright spots
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 1

                # Windows are typically rectangular with aspect ratio 1-4
                if 1 < aspect < 4 and area > 5000:
                    # Check if this bright area is completely blown out
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    region_mean = cv2.mean(gray, mask=mask)[0]

                    if region_mean > 250:  # Completely white = blown window
                        blown_windows += 1
                        total_bright_area += area

            blown_percentage = (total_bright_area / img_area) * 100

            if blown_windows > 2 or blown_percentage > 15:
                issues.append(QAIssue(
                    check_name="Window Exposure",
                    status="fail",
                    message=f"Windows appear BLOWN OUT ({blown_windows} windows, {blown_percentage:.1f}% of image)",
                    action="HDR blending needed - windows should show exterior view"
                ))
            elif blown_windows > 0 or blown_percentage > 5:
                issues.append(QAIssue(
                    check_name="Window Exposure",
                    status="warning",
                    message=f"Some windows may be overexposed ({blown_windows} detected)",
                    action="Check window exposure in Lightroom - may need local adjustment"
                ))
            else:
                issues.append(QAIssue(check_name="Window Exposure", status="pass", message="Windows properly exposed"))
        except Exception:
            issues.append(QAIssue(check_name="Window Exposure", status="info", message="Could not analyze windows"))

        # Step 14: Vertical Lines Check - Perspective correction
        update_progress(14, 18, "Checking vertical lines...")
        try:
            # Use Canny edge detection and Hough lines to find vertical lines
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

            if lines is not None:
                vertical_angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Calculate angle from vertical
                    if x2 - x1 != 0:
                        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        # Vertical lines should be close to 90 or -90 degrees
                        deviation_from_vertical = min(abs(angle - 90), abs(angle + 90), abs(angle - 270))
                        if deviation_from_vertical < 15:  # Within 15 degrees of vertical
                            vertical_angles.append(deviation_from_vertical)

                if vertical_angles:
                    avg_deviation = np.mean(vertical_angles)
                    if avg_deviation > 3:
                        issues.append(QAIssue(
                            check_name="Vertical Lines",
                            status="warning",
                            message=f"Vertical lines appear tilted (avg {avg_deviation:.1f}° off)",
                            action="Apply perspective correction - use Lightroom Transform or Photoshop"
                        ))
                    else:
                        issues.append(QAIssue(check_name="Vertical Lines", status="pass", message="Verticals appear straight"))
                else:
                    issues.append(QAIssue(check_name="Vertical Lines", status="info", message="No strong vertical lines detected"))
            else:
                issues.append(QAIssue(check_name="Vertical Lines", status="info", message="Could not detect lines"))
        except Exception:
            issues.append(QAIssue(check_name="Vertical Lines", status="info", message="Could not analyze perspective"))

        # Step 15: HDR Artifacts - Check for halos around high-contrast edges
        update_progress(15, 18, "Checking for HDR artifacts...")
        try:
            # Detect halos by looking for bright bands along edges
            # Halos appear as light/dark bands along high-contrast boundaries
            edges = cv2.Canny(gray, 100, 200)

            # Dilate edges to create a band around them
            kernel = np.ones((7, 7), np.uint8)
            edge_band = cv2.dilate(edges, kernel, iterations=2)

            # Check for unusual brightness patterns along edges
            edge_mask = edge_band > 0
            if np.sum(edge_mask) > 0:
                edge_brightness = gray[edge_mask]
                non_edge_brightness = gray[~edge_mask]

                # Halos cause edge regions to be abnormally bright compared to surroundings
                edge_mean = np.mean(edge_brightness)
                non_edge_mean = np.mean(non_edge_brightness)
                brightness_ratio = edge_mean / non_edge_mean if non_edge_mean > 0 else 1

                if brightness_ratio > 1.3:
                    issues.append(QAIssue(
                        check_name="HDR Artifacts",
                        status="warning",
                        message=f"Possible HDR halos detected (edge brightness ratio: {brightness_ratio:.2f})",
                        action="Check for halos around windows/doorways - reduce HDR intensity"
                    ))
                else:
                    issues.append(QAIssue(check_name="HDR Artifacts", status="pass", message="No HDR halos detected"))
            else:
                issues.append(QAIssue(check_name="HDR Artifacts", status="info", message="Could not analyze HDR"))
        except Exception:
            issues.append(QAIssue(check_name="HDR Artifacts", status="info", message="Could not check for HDR artifacts"))

        # Step 16: Camera/EXIF Metadata Check
        update_progress(16, 18, "Checking camera metadata...")
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            pil_img = Image.open(file_path)
            exif_data = pil_img._getexif()

            if exif_data:
                exif_info = {}
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_info[tag] = value

                # Check for key metadata
                camera_make = exif_info.get('Make', 'Unknown')
                camera_model = exif_info.get('Model', 'Unknown')
                iso = exif_info.get('ISOSpeedRatings', 'Unknown')
                aperture = exif_info.get('FNumber', 'Unknown')
                focal_length = exif_info.get('FocalLength', 'Unknown')

                metadata_issues = []

                # Flag high ISO (noisy images)
                if isinstance(iso, int) and iso > 1600:
                    metadata_issues.append(f"High ISO ({iso}) may cause noise")

                # Flag unusual aperture for real estate
                if isinstance(aperture, tuple):
                    f_number = aperture[0] / aperture[1] if aperture[1] > 0 else 0
                    if f_number > 0 and (f_number < 5.6 or f_number > 16):
                        metadata_issues.append(f"Unusual aperture (f/{f_number:.1f})")

                camera_info = f"{camera_make} {camera_model}".strip()
                if camera_info and camera_info != "Unknown Unknown":
                    if metadata_issues:
                        issues.append(QAIssue(
                            check_name="Camera Settings",
                            status="warning",
                            message=f"{camera_info} - {'; '.join(metadata_issues)}",
                            action="Review camera settings for optimal image quality"
                        ))
                    else:
                        issues.append(QAIssue(check_name="Camera Settings", status="pass", message=camera_info))
                else:
                    issues.append(QAIssue(check_name="Camera Settings", status="info", message="Camera metadata not found"))
            else:
                issues.append(QAIssue(check_name="Camera Settings", status="info", message="No EXIF data"))
        except Exception:
            issues.append(QAIssue(check_name="Camera Settings", status="info", message="Could not read EXIF data"))

        # Step 17: Filename (skip for batch/Dropbox downloads)
        update_progress(17, 18, "Checking filename...")
        if not skip_filename_check:
            valid_prefixes = ['Photo_', 'Drone_', 'Twilight_', 'TwilightDrone_', 'Lifestyle_', 'PickupPhoto_', 'ReshootPhoto_']
            valid_suffixes = ['Print-', 'Web-']
            has_valid_prefix = any(filename.startswith(p) for p in valid_prefixes)
            has_valid_suffix = any(s in filename for s in valid_suffixes)
            if has_valid_prefix and has_valid_suffix:
                issues.append(QAIssue(check_name="File Naming", status="pass", message="Valid AC naming"))
            else:
                issues.append(QAIssue(check_name="File Naming", status="info", message="Non-standard filename"))

        # Step 18: Room Detection (for cross-training video classification)
        update_progress(18, 18, "Detecting room type...")
        detected_room = "unknown"
        room_confidence = 0.0
        detected_objects = []

        try:
            # Use the small image we already have for room detection
            detected_room, room_confidence = classify_photo_room(file_path, filename, img_small)

            # Get detected objects for learning
            if YOLO_AVAILABLE:
                objects = detect_objects_in_frame(img_small)
                detected_objects = list(objects.keys()) if objects else []

            # Store detection for cross-training video
            folder_path = os.path.dirname(file_path)
            stats_tracker.log_photo_room_detection(
                filename=filename,
                folder_path=folder_path,
                detected_room=detected_room,
                confidence=room_confidence,
                objects_detected=detected_objects
            )

            # Add room detection to issues (info only)
            room_name = ROOM_TYPES.get(detected_room, {}).get('name', detected_room.title())
            if room_confidence > 0.5:
                issues.append(QAIssue(
                    check_name="Room Detection",
                    status="info",
                    message=f"{room_name} ({room_confidence*100:.0f}% confidence)"
                ))
            elif detected_room != "unknown":
                issues.append(QAIssue(
                    check_name="Room Detection",
                    status="info",
                    message=f"Possibly {room_name} (low confidence)"
                ))
        except Exception:
            pass  # Room detection is optional, don't fail QA if it errors

        has_failures = any(i.status == "fail" for i in issues)

        # Track stats
        stats_tracker.increment_stat('total_photos_analyzed')
        issue_count = sum(1 for i in issues if i.status in ('fail', 'warning'))
        stats_tracker.increment_stat('total_issues_found', issue_count)

        return QAReport(filename=filename, file_type="photo", overall_status="fail" if has_failures else "pass",
            issues=issues, metadata={
                'file_path': file_path,  # Store path for thumbnail preview
                'width': width,
                'height': height,
                'detected_room': detected_room,
                'room_confidence': room_confidence,
                'detected_objects': detected_objects
            })

    except Exception as e:
        stats_tracker.increment_stat('total_photos_analyzed')
        stats_tracker.increment_stat('total_issues_found', 1)
        return QAReport(filename=filename, file_type="photo", overall_status="fail",
            issues=[QAIssue(check_name="Analysis", status="fail", message=str(e))], metadata={})

# ============================================================================
# DROPBOX INTEGRATION
# ============================================================================

def extract_zip_photos(zip_path: str) -> List[str]:
    """Extract photos from a ZIP file and return list of image paths"""
    import zipfile

    photo_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    extracted_files = []

    # Create temp directory for extraction
    extract_dir = tempfile.mkdtemp()

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Skip directories and hidden files
                if file_info.is_dir() or file_info.filename.startswith('__MACOSX') or '/.' in file_info.filename:
                    continue

                # Check if it's an image
                ext = os.path.splitext(file_info.filename)[1].lower()
                if ext in photo_extensions:
                    # Extract the file
                    extracted_path = zip_ref.extract(file_info, extract_dir)
                    extracted_files.append(extracted_path)

        return sorted(extracted_files)
    except Exception as e:
        return []


def run_batch_photo_qa(file_paths: List[str], progress_callback=None, skip_filename_check: bool = True) -> List[QAReport]:
    """Run QA on multiple photos with progress updates

    Args:
        file_paths: List of photo file paths to analyze
        progress_callback: Optional callback for progress updates
        skip_filename_check: Skip filename validation for batch (default True since Dropbox can change names)
    """
    reports = []
    total = len(file_paths)

    for idx, file_path in enumerate(file_paths):
        if progress_callback:
            filename = os.path.basename(file_path)
            progress_callback((idx + 1) / total, f"Analyzing {idx + 1} of {total}: {filename[:30]}...")

        report = run_photo_qa(file_path, skip_filename_check=skip_filename_check)
        reports.append(report)

    return reports


def get_photo_thumbnail_base64(file_path: str, max_width: int = 300) -> Optional[str]:
    """Generate a base64 thumbnail for inline display"""
    try:
        import cv2
        import base64

        img = cv2.imread(file_path)
        if img is None:
            return None

        # Resize to thumbnail
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Encode to JPEG base64
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return None


def display_batch_report(reports: List[QAReport]):
    """Display summary of batch QA results with photo previews"""
    passed = sum(1 for r in reports if r.overall_status == "pass")
    failed = sum(1 for r in reports if r.overall_status == "fail")
    total = len(reports)

    # Summary header
    if failed == 0:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(74, 222, 128, 0.15), rgba(74, 222, 128, 0.03));
                    border: 1px solid rgba(74, 222, 128, 0.4); border-radius: 12px; padding: 20px; margin-bottom: 20px; text-align: center;">
            <div style="margin-bottom: 8px;">{icon('pass', 32)}</div>
            <div style="font-size: 18px; font-weight: 600; color: #4ade80;">ALL {total} PHOTOS PASSED</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.02));
                    border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 20px; text-align: center;">
            <div style="margin-bottom: 8px;">{icon('warning', 32)}</div>
            <div style="font-size: 18px; font-weight: 600; color: #fff;">{passed} Passed · <span style="color: #ef4444;">{failed} Failed</span></div>
            <div style="font-size: 13px; color: #a1a1aa; margin-top: 4px;">{total} photos analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    # Show failed photos first with previews
    if failed > 0:
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0 8px;">
                {icon('fail', 18)}
                <span style="color: #fff; font-weight: 600;">Photos Needing Fixes</span>
            </div>
        """, unsafe_allow_html=True)

        fail_idx = 0
        for idx, report in enumerate(reports):
            if report.overall_status == "fail":
                failures = [i for i in report.issues if i.status == "fail"]
                fail_names = ", ".join([i.check_name for i in failures])
                display_name = report.filename

                # Show expander for each failed photo (plain text to avoid rendering issues)
                with st.expander(f"FAILED: {display_name}", expanded=True):
                    # Two columns: photo preview and issues
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        # Show photo with st.image (has built-in click-to-zoom)
                        file_path = report.metadata.get('file_path', '')
                        if file_path and os.path.exists(file_path):
                            st.image(file_path, caption="Click to enlarge", use_container_width=True)
                        else:
                            st.markdown("""
                            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 8px;
                                        height: 150px; display: flex; align-items: center; justify-content: center;">
                                <span style="color: #71717a;">Preview unavailable</span>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"**Issues:** {fail_names}")
                        for issue in failures:
                            st.markdown(f"""
                            <div style="background: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444;
                                        padding: 8px 12px; margin: 8px 0; border-radius: 0 6px 6px 0;">
                                <div style="color: #fff; font-weight: 600; font-size: 13px;">{issue.check_name}</div>
                                <div style="color: #a1a1aa; font-size: 12px; margin-top: 4px;">{issue.message}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            if issue.action:
                                st.markdown(f"""
                                    <div style="display: flex; align-items: flex-start; gap: 6px; margin: 4px 0 4px 16px;">
                                        {icon('lightbulb', 12)}
                                        <span style="color: #4ade80; font-size: 12px;">{issue.action}</span>
                                    </div>
                                """, unsafe_allow_html=True)

                fail_idx += 1

    # Show passed photos (compact)
    if passed > 0:
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0 8px;">
                {icon('pass', 18)}
                <span style="color: #fff; font-weight: 600;">Passed Photos</span>
            </div>
        """, unsafe_allow_html=True)

        passed_reports = [r for r in reports if r.overall_status == "pass"]

        if len(passed_reports) <= 10:
            for pidx, report in enumerate(passed_reports):
                st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 6px; padding: 4px 0;">
                        {icon('check', 14)}
                        <span style="color: #a1a1aa; font-size: 13px;">{report.filename}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 6px; padding: 2px 0;">
                    {icon('check', 14)}
                    <span style="color: #a1a1aa; font-size: 13px;">{len(passed_reports)} photos passed all checks</span>
                </div>
            """, unsafe_allow_html=True)
            with st.expander("View all passed photos"):
                for pidx, report in enumerate(passed_reports):
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 6px; padding: 2px 0;">
                            {icon('check', 14)}
                            <span style="color: #a1a1aa; font-size: 13px;">{report.filename}</span>
                        </div>
                    """, unsafe_allow_html=True)


def download_from_dropbox(share_link: str) -> tuple:
    """
    Download a file from a Dropbox share link.

    Converts share links to direct download URLs:
    - Changes ?dl=0 to ?dl=1
    - Handles various Dropbox URL formats

    Returns: (temp_file_path, filename, error_message)
    """
    try:
        import urllib.parse

        # Clean up the URL
        url = share_link.strip()

        # Extract filename from URL
        parsed = urllib.parse.urlparse(url)
        path_parts = parsed.path.split('/')
        filename = path_parts[-1] if path_parts else "download"

        # Convert to direct download link
        if 'dropbox.com' in url:
            # Remove any existing dl parameter and add dl=1
            if '?' in url:
                base_url = url.split('?')[0]
                params = urllib.parse.parse_qs(parsed.query)
                # Keep rlkey if present (required for access)
                rlkey = params.get('rlkey', [''])[0]
                if rlkey:
                    direct_url = f"{base_url}?rlkey={rlkey}&dl=1"
                else:
                    direct_url = f"{base_url}?dl=1"
            else:
                direct_url = f"{url}?dl=1"

            # Also try raw=1 format for better compatibility
            direct_url = direct_url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
            direct_url = direct_url.replace('?dl=1', '').replace('&dl=1', '')
            if '?rlkey=' in direct_url:
                direct_url = direct_url  # Keep rlkey for authenticated links
        else:
            return None, None, "Not a valid Dropbox link"

        # Download the file
        response = requests.get(direct_url, stream=True, timeout=300)

        if response.status_code != 200:
            # Try alternative URL format
            alt_url = share_link.replace('?dl=0', '?dl=1').replace('&dl=0', '&dl=1')
            if 'dl=1' not in alt_url:
                alt_url = alt_url + ('&dl=1' if '?' in alt_url else '?dl=1')

            response = requests.get(alt_url, stream=True, timeout=300)

            if response.status_code != 200:
                return None, None, f"HTTP {response.status_code} - Could not download file"

        # Get filename from content-disposition header if available
        if 'content-disposition' in response.headers:
            import re
            cd = response.headers['content-disposition']
            fname_match = re.findall('filename="?([^";\n]+)"?', cd)
            if fname_match:
                filename = fname_match[0]

        # Determine file extension
        ext = os.path.splitext(filename)[1].lower()
        if not ext:
            content_type = response.headers.get('content-type', '')
            if 'video' in content_type:
                ext = '.mp4'
            elif 'image' in content_type:
                ext = '.jpg'
            filename = filename + ext

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            return tmp.name, filename, None

    except requests.Timeout:
        return None, None, "Download timed out - file may be too large"
    except Exception as e:
        return None, None, str(e)


# ============================================================================
# STREAMLIT UI
# ============================================================================

def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS"""
    if seconds is None:
        return ""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:05.2f}"


def format_timestamp_short(seconds: float) -> str:
    """Format seconds to M:SS for display"""
    if seconds is None:
        return ""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def create_timeline_markers_html(duration: float, markers: list, selected_idx: int = 0) -> str:
    """
    Generate HTML for a timeline bar with issue markers.
    This is displayed BELOW the Streamlit video player as a visual reference.

    markers: list of dicts with {timestamp, color, label, index}
    """
    if not markers or duration <= 0:
        return ""

    # Build marker HTML
    markers_html = ""
    for m in markers:
        pos_pct = (m['timestamp'] / duration * 100) if duration > 0 else 0
        pos_pct = max(1, min(99, pos_pct))
        is_sel = m['index'] == selected_idx
        size = 16 if is_sel else 12
        border = "2px solid #fff" if is_sel else "1px solid rgba(255,255,255,0.4)"
        z = 30 if is_sel else 20
        glow = f"0 0 10px {m['color']}" if is_sel else f"0 0 5px {m['color']}60"

        markers_html += f'''
            <div style="position: absolute; left: {pos_pct}%; top: 50%; transform: translate(-50%, -50%);
                        width: {size}px; height: {size}px; background: {m['color']}; border-radius: 50%;
                        border: {border}; z-index: {z}; box-shadow: {glow};
                        transition: transform 0.15s;"
                 title="{m['label']} @ {int(m['timestamp']//60)}:{int(m['timestamp']%60):02d}">
            </div>
        '''

    html = f'''
    <div style="position: relative; background: linear-gradient(90deg, #1a1a1a, #252525, #1a1a1a);
                border-radius: 10px; height: 36px; border: 1px solid #333; margin: 4px 0;">
        <!-- Track line -->
        <div style="position: absolute; top: 50%; left: 16px; right: 16px; height: 4px;
                    background: linear-gradient(90deg, #333 0%, #444 50%, #333 100%); border-radius: 2px;
                    transform: translateY(-50%);">
        </div>
        <!-- Time labels -->
        <div style="position: absolute; left: 16px; bottom: 4px; color: #52525b; font-size: 10px; font-family: monospace;">0:00</div>
        <div style="position: absolute; right: 16px; bottom: 4px; color: #52525b; font-size: 10px; font-family: monospace;">{int(duration//60)}:{int(duration%60):02d}</div>
        <!-- Markers container -->
        <div style="position: absolute; left: 16px; right: 16px; top: 0; bottom: 0;">
            {markers_html}
        </div>
    </div>
    '''
    return html


def display_video_review_interface(report: QAReport, video_path: str = None, show_feedback: bool = True):
    """
    Compact Frame.io-style video review interface:
    - Custom video player with issue markers ON the scrubber bar
    - Click markers to jump to that timestamp
    - Issue details panel below
    """
    duration = report.metadata.get('duration', 0)

    # Count results
    passes = sum(1 for i in report.issues if i.status == "pass")
    failures = sum(1 for i in report.issues if i.status == "fail")
    warnings = sum(1 for i in report.issues if i.status == "warning")
    total = passes + failures + warnings
    is_passing = failures == 0

    # Separate issues by type
    timeline_issues = [i for i in report.issues if i.timestamp_start is not None and i.status in ('fail', 'warning')]
    general_issues = [i for i in report.issues if i.timestamp_start is None and i.status in ('fail', 'warning')]
    passed_issues = [i for i in report.issues if i.status == "pass"]

    # Sort timeline issues by timestamp
    timeline_issues.sort(key=lambda x: x.timestamp_start or 0)

    # Unique key for this report's selection state
    report_key = f"video_review_{hash(report.filename)}"
    if report_key not in st.session_state:
        st.session_state[report_key] = 0

    # Helper to get/set selected index
    def get_selected():
        return st.session_state.get(report_key, 0) if timeline_issues else None

    def set_selected(idx):
        st.session_state[report_key] = idx

    # Categorize issues by type for color coding
    def get_issue_category(check_name):
        check_lower = check_name.lower()
        if any(x in check_lower for x in ['audio', 'sound', 'music', 'silence', 'clip']):
            return 'audio'
        elif any(x in check_lower for x in ['black', 'frame', 'drop', 'freeze', 'flash']):
            return 'critical'
        elif any(x in check_lower for x in ['color', 'exposure', 'white balance', 'saturation', 'log']):
            return 'color'
        elif any(x in check_lower for x in ['motion', 'shake', 'stabiliz']):
            return 'motion'
        else:
            return 'general'

    category_colors = {
        'critical': {'bg': '#ef4444', 'label': 'Critical'},
        'audio': {'bg': '#f59e0b', 'label': 'Audio'},
        'color': {'bg': '#3b82f6', 'label': 'Color'},
        'motion': {'bg': '#a855f7', 'label': 'Motion'},
        'general': {'bg': '#71717a', 'label': 'General'},
    }

    # =============================================
    # COMPACT HEADER
    # =============================================
    score = int(passes/total*100) if total > 0 else (100 if is_passing else 0)
    if is_passing:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(74, 222, 128, 0.12), rgba(74, 222, 128, 0.02));
                    border: 1px solid rgba(74, 222, 128, 0.3); border-radius: 12px; padding: 12px 16px; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="width: 36px; height: 36px; background: rgba(74, 222, 128, 0.2); border-radius: 50%;
                                display: flex; align-items: center; justify-content: center; color: #4ade80; font-size: 18px; font-weight: 700;">✓</div>
                    <div>
                        <div style="font-size: 16px; font-weight: 600; color: #4ade80;">VIDEO PASSED</div>
                        <div style="font-size: 11px; color: #71717a;">{passes}/{total} checks{f' · {warnings} warning(s)' if warnings > 0 else ''}</div>
                    </div>
                </div>
                <div style="font-size: 24px; font-weight: 700; color: #4ade80;">{score}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.02));
                    border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 12px 16px; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="width: 36px; height: 36px; background: rgba(239, 68, 68, 0.2); border-radius: 50%;
                                display: flex; align-items: center; justify-content: center; color: #ef4444; font-size: 18px; font-weight: 700;">!</div>
                    <div>
                        <div style="font-size: 16px; font-weight: 600; color: #ef4444;">NEEDS ATTENTION</div>
                        <div style="font-size: 11px; color: #71717a;">{failures} issue(s) · {warnings} warning(s)</div>
                    </div>
                </div>
                <div style="font-size: 24px; font-weight: 700; color: #ef4444;">{score}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =============================================
    # VIDEO PLAYER + ISSUE TIMELINE
    # =============================================
    selected_idx = get_selected()

    if video_path and os.path.exists(video_path):
        # Standard Streamlit video player (handles large files)
        st.video(video_path)

        # Build markers list for the timeline
        markers = []
        for i, issue in enumerate(timeline_issues):
            category = get_issue_category(issue.check_name)
            markers.append({
                'timestamp': issue.timestamp_start,
                'color': category_colors[category]['bg'],
                'label': issue.check_name,
                'index': i
            })

        # Render timeline with markers below video (if there are issues)
        if markers and duration > 0:
            timeline_html = create_timeline_markers_html(duration, markers, selected_idx or 0)
            st.markdown(timeline_html, unsafe_allow_html=True)

        # Video info and legend bar
        duration_fmt = report.metadata.get('duration_formatted', format_timestamp_short(duration))
        resolution = f"{report.metadata.get('width', '?')}x{report.metadata.get('height', '?')}"
        fps = report.metadata.get('fps', '?')

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0 8px;">
            <div style="display: flex; gap: 12px; font-size: 10px; color: #71717a;">
                <span>{resolution} · {fps}fps · {duration_fmt}</span>
                <span style="color: #52525b;">|</span>
                <span style="display: flex; align-items: center; gap: 4px;">{color_dot('#ef4444', 6)}Critical</span>
                <span style="display: flex; align-items: center; gap: 4px;">{color_dot('#f59e0b', 6)}Audio</span>
                <span style="display: flex; align-items: center; gap: 4px;">{color_dot('#3b82f6', 6)}Color</span>
                <span style="display: flex; align-items: center; gap: 4px;">{color_dot('#a855f7', 6)}Motion</span>
            </div>
            <span style="color: #52525b; font-size: 10px;">{len(timeline_issues)} issues found</span>
        </div>
        """, unsafe_allow_html=True)

    # =============================================
    # CLICKABLE ISSUE BUTTONS (select which issue to view)
    # =============================================
    if timeline_issues:
        num_issues = len(timeline_issues)
        if num_issues <= 6:
            cols = st.columns(num_issues)
        else:
            cols = st.columns(6)

        for i, issue in enumerate(timeline_issues):
            col_idx = i % 6 if num_issues > 6 else i
            category = get_issue_category(issue.check_name)
            color = category_colors[category]['bg']
            timestamp = format_timestamp_short(issue.timestamp_start)
            is_selected = (i == selected_idx)

            with cols[col_idx]:
                btn_type = "primary" if is_selected else "secondary"
                if st.button(
                    f"{timestamp}",
                    key=f"issue_btn_{report_key}_{i}",
                    use_container_width=True,
                    type=btn_type
                ):
                    set_selected(i)
                    st.rerun()

    # =============================================
    # SELECTED ISSUE DETAIL PANEL
    # =============================================
    if timeline_issues and selected_idx is not None and selected_idx < len(timeline_issues):
            issue = timeline_issues[selected_idx]
            category = get_issue_category(issue.check_name)
            cat_info = category_colors[category]
            timestamp = format_timestamp_short(issue.timestamp_start)

            # Severity styling
            if issue.status == 'fail':
                sev_bg = "rgba(239, 68, 68, 0.08)"
                sev_border = "rgba(239, 68, 68, 0.25)"
                sev_text = "#ef4444"
                sev_label = "FAIL"
            else:
                sev_bg = "rgba(245, 158, 11, 0.08)"
                sev_border = "rgba(245, 158, 11, 0.25)"
                sev_text = "#f59e0b"
                sev_label = "WARNING"

            st.markdown(f"""
            <div style="background: {sev_bg}; border: 1px solid {sev_border}; border-radius: 10px;
                        padding: 14px; margin-top: 12px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    <span style="background: {cat_info['bg']}; color: #fff; padding: 3px 8px; border-radius: 12px;
                                 font-size: 10px; font-weight: 600;">{cat_info['label'].upper()}</span>
                    <span style="color: {sev_text}; font-size: 10px; font-weight: 600;">{sev_label}</span>
                    <span style="color: #52525b; font-size: 10px; margin-left: auto;">Issue {selected_idx + 1} of {len(timeline_issues)}</span>
                </div>
                <div style="color: #fff; font-weight: 600; font-size: 14px; margin-bottom: 6px;">{issue.check_name}</div>
                <div style="color: #a1a1aa; font-size: 12px; line-height: 1.5; margin-bottom: 10px;">{issue.message}</div>
                <div style="display: flex; align-items: center; gap: 8px; color: #71717a; font-size: 11px;">
                    <span style="background: #27272a; padding: 4px 8px; border-radius: 6px; font-family: monospace;">@ {timestamp}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show thumbnail if available
            if issue.preview_image:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <div style="margin-top: 8px;">
                        <img src="data:image/jpeg;base64,{issue.preview_image}"
                             style="width: 100%; border-radius: 6px; border: 2px solid {cat_info['bg']};">
                        <div style="text-align: center; color: #52525b; font-size: 10px; margin-top: 4px;">Frame @ {timestamp}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if issue.action:
                        st.markdown(f"""
                        <div style="background: rgba(123, 140, 222, 0.08); border: 1px solid rgba(123, 140, 222, 0.2);
                                    border-radius: 8px; padding: 10px; margin-top: 8px;">
                            <div style="color: #7B8CDE; font-size: 10px; font-weight: 600; margin-bottom: 4px;">HOW TO FIX</div>
                            <div style="color: #d4d4d8; font-size: 12px;">{issue.action}</div>
                        </div>
                        """, unsafe_allow_html=True)
            elif issue.action:
                st.markdown(f"""
                <div style="background: rgba(123, 140, 222, 0.08); border: 1px solid rgba(123, 140, 222, 0.2);
                            border-radius: 8px; padding: 10px; margin-top: 8px;">
                    <div style="color: #7B8CDE; font-size: 10px; font-weight: 600; margin-bottom: 4px;">HOW TO FIX</div>
                    <div style="color: #d4d4d8; font-size: 12px;">{issue.action}</div>
                </div>
                """, unsafe_allow_html=True)

            # Navigation buttons
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if selected_idx > 0:
                    if st.button("← Previous", key=f"prev_{report_key}", use_container_width=True):
                        set_selected(selected_idx - 1)
                        st.rerun()
            with nav_col3:
                if selected_idx < len(timeline_issues) - 1:
                    if st.button("Next →", key=f"next_{report_key}", use_container_width=True):
                        set_selected(selected_idx + 1)
                        st.rerun()

    # =============================================
    # GENERAL ISSUES - COMPACT VIEW (one line per issue)
    # =============================================
    if general_issues:
        st.markdown(f"""
        <div style="margin-top: 24px; background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #1d1d1f;">
                {icon('info', 16)}
                <span style="color: #fff; font-weight: 600; font-size: 14px;">Issues Summary</span>
                <span style="color: #71717a; font-size: 11px;">({len(general_issues)} items)</span>
            </div>
        """, unsafe_allow_html=True)

        # Build compact issue rows
        for issue in general_issues:
            severity_color = "#ef4444" if issue.status == 'fail' else "#f59e0b"
            severity_label = "FAIL" if issue.status == 'fail' else "WARN"

            # Truncate long messages
            short_msg = issue.message[:80] + "..." if len(issue.message) > 80 else issue.message
            fix_hint = f" → {issue.action[:60]}..." if issue.action and len(issue.action) > 60 else (f" → {issue.action}" if issue.action else "")

            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.03);">
                <span style="background: {severity_color}; color: #000; font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 4px; min-width: 32px; text-align: center;">{severity_label}</span>
                <span style="color: #fff; font-weight: 500; font-size: 13px; min-width: 120px;">{issue.check_name}</span>
                <span style="color: #a1a1aa; font-size: 12px; flex: 1;">{short_msg}</span>
                <span style="color: #7B8CDE; font-size: 11px;">{fix_hint}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Expandable details for those who want more info
        with st.expander("View detailed fixes", expanded=False):
            for issue in general_issues:
                if issue.action:
                    st.markdown(f"""
                    <div style="padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: #fff; font-weight: 500;">{issue.check_name}:</span>
                        <span style="color: #a1a1aa; font-size: 13px;"> {issue.action}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # =============================================
    # PASSED CHECKS (collapsed)
    # =============================================
    if passed_issues:
        with st.expander(f"Passed Checks ({len(passed_issues)})", expanded=False):
            for issue in passed_issues:
                st.markdown(f"""
                <div style="display: flex; align-items: center; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <span style="color: #4ade80; margin-right: 10px;">✓</span>
                    <span style="color: #fff; font-weight: 500;">{issue.check_name}</span>
                    <span style="color: #71717a; margin-left: 12px; font-size: 12px;">{issue.message}</span>
                </div>
                """, unsafe_allow_html=True)


def display_video_timeline_report(report: QAReport, video_path: str = None, show_feedback: bool = True):
    """
    Display video QA report with interactive timeline visualization.
    Shows clickable markers at issue timestamps with thumbnails.
    General issues (no timestamp) are shown below the timeline.
    """
    duration = report.metadata.get('duration', 0)

    # Count results
    passes = sum(1 for i in report.issues if i.status == "pass")
    failures = sum(1 for i in report.issues if i.status == "fail")
    warnings = sum(1 for i in report.issues if i.status == "warning")
    total = passes + failures + warnings
    is_passing = failures == 0

    # Separate issues with timestamps vs general issues
    timeline_issues = [i for i in report.issues if i.timestamp_start is not None and i.status in ('fail', 'warning')]
    general_issues = [i for i in report.issues if i.timestamp_start is None and i.status in ('fail', 'warning')]
    passed_issues = [i for i in report.issues if i.status == "pass"]

    # =============================================
    # HEADER WITH SCORE
    # =============================================
    if is_passing:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(74, 222, 128, 0.15), rgba(74, 222, 128, 0.03));
                    border: 1px solid rgba(74, 222, 128, 0.4); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;
                    display: flex; align-items: center; gap: 16px;">
            <div>{icon('pass', 32)}</div>
            <div>
                <div style="font-size: 16px; font-weight: 600; color: #4ade80;">PASSED</div>
                <div style="font-size: 12px; color: #a1a1aa;">{passes}/{total} checks{f' - {warnings} warning(s)' if warnings > 0 else ''}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.03));
                    border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;
                    display: flex; align-items: center; gap: 16px;">
            <div>{icon('fail', 32)}</div>
            <div>
                <div style="font-size: 16px; font-weight: 600; color: #ef4444;">FAILED</div>
                <div style="font-size: 12px; color: #a1a1aa;">{failures} issue(s) must be fixed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # File info
    duration_fmt = report.metadata.get('duration_formatted', '')
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 16px;">
            {icon('file', 14)}
            <span style="color: #a1a1aa; font-size: 13px;">{report.filename}{f" - {duration_fmt}" if duration_fmt else ""}</span>
        </div>
    """, unsafe_allow_html=True)

    # =============================================
    # INTERACTIVE TIMELINE (if there are timeline issues)
    # =============================================
    if timeline_issues and duration > 0:
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 20px 0 12px;">
                {icon('timeline', 18)}
                <span style="color: #fff; font-weight: 600;">Timeline View</span>
                <span style="color: #71717a; font-size: 12px; margin-left: 8px;">Click markers to see issue details</span>
            </div>
        """, unsafe_allow_html=True)

        # Build timeline visualization with markers
        # Calculate marker positions as percentages
        markers_html = ""
        for i, issue in enumerate(timeline_issues):
            position_pct = (issue.timestamp_start / duration) * 100 if duration > 0 else 0
            position_pct = max(0, min(98, position_pct))  # Keep within bounds

            # Color based on status
            marker_color = "#ef4444" if issue.status == "fail" else "#f59e0b"
            marker_border = "rgba(239, 68, 68, 0.5)" if issue.status == "fail" else "rgba(245, 158, 11, 0.5)"

            markers_html += f"""
                <div class="timeline-marker" data-issue="{i}" style="left: {position_pct}%;
                    position: absolute; top: 50%; transform: translate(-50%, -50%);
                    width: 14px; height: 14px; background: {marker_color}; border-radius: 50%;
                    border: 2px solid {marker_border}; cursor: pointer; z-index: 10;
                    box-shadow: 0 0 8px {marker_color}40;"
                    title="{issue.check_name} @ {format_timestamp(issue.timestamp_start)}">
                </div>
            """

        # Timeline bar
        st.markdown(f"""
        <div style="position: relative; background: #1d1d1f; border-radius: 8px; height: 24px; margin: 16px 0;">
            <!-- Timeline track -->
            <div style="position: absolute; top: 50%; left: 8px; right: 8px; height: 4px;
                        background: linear-gradient(90deg, #333 0%, #444 100%); border-radius: 2px;
                        transform: translateY(-50%);">
            </div>

            <!-- Progress/duration labels -->
            <div style="position: absolute; left: 8px; top: -18px; color: #71717a; font-size: 10px;">0:00</div>
            <div style="position: absolute; right: 8px; top: -18px; color: #71717a; font-size: 10px;">{format_timestamp(duration)}</div>

            <!-- Markers -->
            <div style="position: absolute; left: 8px; right: 8px; top: 0; bottom: 0;">
                {markers_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Issue cards for timeline issues
        st.markdown(f"""
            <div style="margin-top: 16px;">
        """, unsafe_allow_html=True)

        for i, issue in enumerate(timeline_issues):
            issue_color = "#ef4444" if issue.status == "fail" else "#f59e0b"
            issue_bg = "rgba(239, 68, 68, 0.1)" if issue.status == "fail" else "rgba(245, 158, 11, 0.1)"
            issue_border = "rgba(239, 68, 68, 0.3)" if issue.status == "fail" else "rgba(245, 158, 11, 0.3)"
            status_icon = icon('fail', 16) if issue.status == "fail" else icon('warning', 16)

            # Issue card header
            expander_title = f"{issue.check_name} @ {format_timestamp(issue.timestamp_start)}"
            with st.expander(expander_title, expanded=False):
                # Two-column layout: thumbnail on left, details on right
                if issue.preview_image:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f'''
                            <div style="border: 2px solid {issue_color}; border-radius: 8px; overflow: hidden;">
                                <img src="data:image/jpeg;base64,{issue.preview_image}" style="width: 100%; display: block;">
                            </div>
                            <div style="text-align: center; margin-top: 8px;">
                                <span style="color: #71717a; font-size: 11px;">@ {format_timestamp(issue.timestamp_start)}</span>
                            </div>
                        ''', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"**Issue:** {issue.message}")
                        if issue.action:
                            st.markdown(f"""
                                <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                                            border-radius: 8px; padding: 12px; margin-top: 8px; display: flex; align-items: flex-start; gap: 8px;">
                                    {icon('lightbulb', 16)}
                                    <span style="color: #fff; font-size: 13px;">{issue.action}</span>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**Issue:** {issue.message}")
                    if issue.action:
                        st.markdown(f"""
                            <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                                        border-radius: 8px; padding: 12px; margin-top: 8px; display: flex; align-items: flex-start; gap: 8px;">
                                {icon('lightbulb', 16)}
                                <span style="color: #fff; font-size: 13px;">{issue.action}</span>
                            </div>
                        """, unsafe_allow_html=True)

                # Feedback buttons
                if show_feedback:
                    st.markdown("---")
                    st.caption("Was this check accurate?")
                    render_feedback_buttons(report, issue, i)

        st.markdown("</div>", unsafe_allow_html=True)

    # =============================================
    # GENERAL ISSUES (no timestamp)
    # =============================================
    if general_issues:
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 24px 0 8px;">
                {icon('info', 18)}
                <span style="color: #fff; font-weight: 600;">General Issues</span>
            </div>
        """, unsafe_allow_html=True)

        for i, issue in enumerate(general_issues):
            status_icon = 'fail' if issue.status == 'fail' else 'warning'
            expander_title = issue.check_name

            with st.expander(expander_title, expanded=False):
                st.markdown(f"**Issue:** {issue.message}")
                if issue.expected:
                    st.markdown(f"**Expected:** {issue.expected}")
                if issue.found:
                    st.markdown(f"**Found:** {issue.found}")
                if issue.action:
                    st.markdown(f"""
                        <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                                    border-radius: 8px; padding: 12px; margin-top: 8px; display: flex; align-items: flex-start; gap: 8px;">
                            {icon('lightbulb', 16)}
                            <span style="color: #fff; font-size: 13px;">{issue.action}</span>
                        </div>
                    """, unsafe_allow_html=True)

                if show_feedback:
                    st.markdown("---")
                    st.caption("Was this check accurate?")
                    render_feedback_buttons(report, issue, i + 1000)

    # =============================================
    # PASSED CHECKS
    # =============================================
    if passed_issues:
        with st.expander(f"Passed ({len(passed_issues)} checks)", expanded=False):
            for pidx, issue in enumerate(passed_issues):
                st.markdown(f"""
                <div style="display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {icon('check', 16)}
                    <span style="color: #fff; font-weight: 500; margin-left: 10px;">{issue.check_name}</span>
                    <span style="color: #a1a1aa; margin-left: 12px; font-size: 12px;">{issue.message}</span>
                </div>
                """, unsafe_allow_html=True)

                if show_feedback:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("Correct", key=f"tl_pass_ok_{report.filename}_{issue.check_name}_{pidx}"):
                            feedback_db.add_feedback(report.filename, report.file_type, issue.check_name,
                                                    issue.status, issue.message, 'correct')
                            st.success("Noted")
                    with col2:
                        if st.button("Should Fail", key=f"tl_pass_wrong_{report.filename}_{issue.check_name}_{pidx}"):
                            feedback_db.add_feedback(report.filename, report.file_type, issue.check_name,
                                                    issue.status, issue.message, 'false_negative')
                            st.warning("Noted!")

    # Info items
    info_items = [i for i in report.issues if i.status == "info"]
    if info_items:
        for issue in info_items:
            st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 6px; margin: 4px 0;">
                    {icon('info', 14)}
                    <span style="color: #a1a1aa; font-size: 12px;">{issue.check_name}: {issue.message}</span>
                </div>
            """, unsafe_allow_html=True)


def render_feedback_buttons(report: QAReport, issue: QAIssue, idx: int):
    """Render feedback buttons for a specific issue"""
    col1, col2, col3 = st.columns([1, 1, 1])

    # Create unique keys for this issue
    key_base = f"{report.filename}_{issue.check_name}_{idx}"

    with col1:
        if st.button("Correct", key=f"correct_{key_base}", use_container_width=True):
            feedback_db.add_feedback(
                report.filename, report.file_type, issue.check_name,
                issue.status, issue.message, 'correct'
            )
            st.success("Thanks! Feedback recorded.")

    with col2:
        if issue.status in ['fail', 'warning']:
            btn_label = "Actually OK"
            feedback_type = 'false_positive'
        else:
            btn_label = "Should Fail"
            feedback_type = 'false_negative'

        if st.button(btn_label, key=f"wrong_{key_base}", use_container_width=True):
            feedback_db.add_feedback(
                report.filename, report.file_type, issue.check_name,
                issue.status, issue.message, feedback_type
            )
            st.warning(f"Got it! This will help calibrate the {issue.check_name} threshold.")

    with col3:
        if st.button("Add Note", key=f"note_{key_base}", use_container_width=True):
            st.session_state[f"show_note_{key_base}"] = True

    # Show note input if requested
    if st.session_state.get(f"show_note_{key_base}", False):
        note = st.text_input("Add a note:", key=f"note_input_{key_base}")
        if st.button("Save Note", key=f"save_note_{key_base}"):
            feedback_db.add_feedback(
                report.filename, report.file_type, issue.check_name,
                issue.status, issue.message, 'note', note
            )
            st.success("Note saved!")
            st.session_state[f"show_note_{key_base}"] = False

def display_report(report: QAReport, show_feedback: bool = True):
    """Display QA report in Streamlit - Clean, at-a-glance design with feedback options"""

    # Count results
    passes = sum(1 for i in report.issues if i.status == "pass")
    failures = sum(1 for i in report.issues if i.status == "fail")
    warnings = sum(1 for i in report.issues if i.status == "warning")
    total = passes + failures + warnings

    # Determine overall verdict - PASS only if NO failures
    is_passing = failures == 0

    # =============================================
    # COMPACT HEADER WITH SCORE
    # =============================================
    if is_passing:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(74, 222, 128, 0.15), rgba(74, 222, 128, 0.03));
                    border: 1px solid rgba(74, 222, 128, 0.4); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;
                    display: flex; align-items: center; gap: 16px;">
            <div>{icon('pass', 32)}</div>
            <div>
                <div style="font-size: 16px; font-weight: 600; color: #4ade80;">PASSED</div>
                <div style="font-size: 12px; color: #a1a1aa;">{passes}/{total} checks{f' · {warnings} warning(s)' if warnings > 0 else ''}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.03));
                    border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;
                    display: flex; align-items: center; gap: 16px;">
            <div>{icon('fail', 32)}</div>
            <div>
                <div style="font-size: 16px; font-weight: 600; color: #ef4444;">FAILED</div>
                <div style="font-size: 12px; color: #a1a1aa;">{failures} issue(s) must be fixed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # File info
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 16px;">
            {icon('file', 14)}
            <span style="color: #a1a1aa; font-size: 13px;">{report.filename}{f" · {report.metadata['duration_formatted']}" if report.metadata.get('duration_formatted') else ""}</span>
        </div>
    """, unsafe_allow_html=True)

    # =============================================
    # FAILED CHECKS - Clean list with expand + feedback
    # =============================================
    if failures > 0:
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0 8px;">
                {icon('fail', 18)}
                <span style="color: #fff; font-weight: 600;">Issues to Fix</span>
            </div>
        """, unsafe_allow_html=True)
        fail_idx = 0
        for issue in report.issues:
            if issue.status == "fail":
                # Add timestamp to expander title if available
                expander_title = issue.check_name
                if issue.timestamp_start:
                    expander_title += f" @ {format_timestamp(issue.timestamp_start)}"

                with st.expander(expander_title, expanded=False):
                    st.markdown(f"**Issue:** {issue.message}")
                    if issue.expected:
                        st.markdown(f"**Expected:** {issue.expected}")
                    if issue.found:
                        st.markdown(f"**Found:** {issue.found}")
                    if issue.timestamp_start:
                        st.markdown(f"**Timestamp:** {format_timestamp(issue.timestamp_start)}")

                    # Show preview image if available
                    if issue.preview_image:
                        st.markdown("**Preview:**")
                        st.markdown(f'''
                            <div style="border: 2px solid #7B8CDE; border-radius: 8px; overflow: hidden; margin: 8px 0;">
                                <img src="data:image/jpeg;base64,{issue.preview_image}" style="width: 100%; display: block;">
                            </div>
                        ''', unsafe_allow_html=True)

                    if issue.action:
                        st.markdown(f"""
                            <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                                        border-radius: 8px; padding: 12px; margin-top: 8px; display: flex; align-items: flex-start; gap: 8px;">
                                {icon('lightbulb', 16)}
                                <span style="color: #fff; font-size: 13px;">{issue.action}</span>
                            </div>
                        """, unsafe_allow_html=True)

                    # Feedback buttons
                    if show_feedback:
                        st.markdown("---")
                        st.caption("Was this check accurate?")
                        render_feedback_buttons(report, issue, fail_idx)
                fail_idx += 1

    # =============================================
    # WARNINGS + feedback
    # =============================================
    if warnings > 0:
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0 8px;">
                {icon('warning', 18)}
                <span style="color: #fff; font-weight: 600;">Warnings</span>
            </div>
        """, unsafe_allow_html=True)
        warn_idx = 0
        for issue in report.issues:
            if issue.status == "warning":
                # Add timestamp to expander title if available
                expander_title = issue.check_name
                if issue.timestamp_start:
                    expander_title += f" @ {format_timestamp(issue.timestamp_start)}"

                with st.expander(expander_title, expanded=False):
                    st.markdown(issue.message)

                    # Show preview image if available
                    if issue.preview_image:
                        st.markdown("**Preview:**")
                        st.markdown(f'''
                            <div style="border: 2px solid #f59e0b; border-radius: 8px; overflow: hidden; margin: 8px 0;">
                                <img src="data:image/jpeg;base64,{issue.preview_image}" style="width: 100%; display: block;">
                            </div>
                        ''', unsafe_allow_html=True)

                    if issue.action:
                        st.markdown(f"""
                            <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                                        border-radius: 8px; padding: 12px; margin-top: 8px; display: flex; align-items: flex-start; gap: 8px;">
                                {icon('lightbulb', 16)}
                                <span style="color: #fff; font-size: 13px;">{issue.action}</span>
                            </div>
                        """, unsafe_allow_html=True)

                    # Feedback buttons
                    if show_feedback:
                        st.markdown("---")
                        st.caption("Was this check accurate?")
                        render_feedback_buttons(report, issue, warn_idx + 1000)
                warn_idx += 1

    # =============================================
    # PASSED CHECKS - Green checkmarks in vertical dropdown + feedback
    # =============================================
    if passes > 0:
        with st.expander(f"Passed ({passes} checks)", expanded=False):
            passed_issues = [i for i in report.issues if i.status == "pass"]
            for pidx, issue in enumerate(passed_issues):
                st.markdown(f"""
                <div style="display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {icon('check', 16)}
                    <span style="color: #fff; font-weight: 500; margin-left: 10px;">{issue.check_name}</span>
                    <span style="color: #a1a1aa; margin-left: 12px; font-size: 12px;">{issue.message}</span>
                </div>
                """, unsafe_allow_html=True)

                # Compact feedback for passed items
                if show_feedback:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("Correct", key=f"pass_ok_{report.filename}_{issue.check_name}_{pidx}"):
                            feedback_db.add_feedback(report.filename, report.file_type, issue.check_name,
                                                    issue.status, issue.message, 'correct')
                            st.success("Noted")
                    with col2:
                        if st.button("Should Fail", key=f"pass_wrong_{report.filename}_{issue.check_name}_{pidx}"):
                            feedback_db.add_feedback(report.filename, report.file_type, issue.check_name,
                                                    issue.status, issue.message, 'false_negative')
                            st.warning("Noted!")

    # Info items
    info_items = [i for i in report.issues if i.status == "info"]
    if info_items:
        for issue in info_items:
            st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 6px; margin: 4px 0;">
                    {icon('info', 14)}
                    <span style="color: #a1a1aa; font-size: 12px;">{issue.check_name}: {issue.message}</span>
                </div>
            """, unsafe_allow_html=True)

# ============================================================================
# AUTO SORT - Footage Organization & XML Export
# ============================================================================

# Room type definitions with visual feature keywords
ROOM_TYPES = {
    "exterior_front": {
        "name": "Front Exterior",
        "icon": "room_exterior",
        "keywords": ["front", "exterior", "curb", "driveway", "facade", "entrance"],
        "features": ["sky", "roof", "windows", "door", "garage_door", "landscaping"]
    },
    "exterior_rear": {
        "name": "Rear Exterior",
        "icon": "room_exterior_rear",
        "keywords": ["rear", "back_exterior", "backside"],
        "features": ["sky", "roof", "back_windows", "deck", "patio_door", "back_of_house"]
    },
    "entry": {
        "name": "Entryway",
        "icon": "room_entry",
        "keywords": ["entry", "foyer", "front", "welcome"],
        "features": ["door", "narrow_space", "hallway", "stairs_visible"]
    },
    "living_room": {
        "name": "Living Room",
        "icon": "room_living",
        "keywords": ["living", "great", "family", "main"],
        "features": ["sofa", "couch", "tv", "fireplace", "large_windows", "open_space"]
    },
    "kitchen": {
        "name": "Kitchen",
        "icon": "room_kitchen",
        "keywords": ["kitchen", "cook"],
        "features": ["cabinets", "countertop", "appliances", "island", "sink", "stove"]
    },
    "dining": {
        "name": "Dining Room",
        "icon": "room_dining",
        "keywords": ["dining", "eat"],
        "features": ["table", "chairs", "chandelier", "formal"]
    },
    "primary_bedroom": {
        "name": "Primary Bedroom",
        "icon": "room_primary_bed",
        "keywords": ["master", "primary", "main_bed"],
        "features": ["large_bed", "ensuite_visible", "spacious"]
    },
    "bedroom": {
        "name": "Bedroom",
        "icon": "room_bedroom",
        "keywords": ["bed", "guest", "room"],
        "features": ["bed", "dresser", "closet", "window"]
    },
    "bathroom": {
        "name": "Bathroom",
        "icon": "room_bathroom",
        "keywords": ["bath", "powder", "half"],
        "features": ["toilet", "sink", "vanity", "shower", "tub", "tile", "mirror"]
    },
    "office": {
        "name": "Office",
        "icon": "room_office",
        "keywords": ["office", "study", "den", "work"],
        "features": ["desk", "shelves", "books", "computer"]
    },
    "front_yard": {
        "name": "Front Yard",
        "icon": "room_yard",
        "keywords": ["front_yard", "front_lawn", "front_landscape"],
        "features": ["grass", "trees", "plants", "fence", "sky", "walkway"]
    },
    "backyard": {
        "name": "Backyard",
        "icon": "room_backyard",
        "keywords": ["back", "backyard", "rear_yard"],
        "features": ["grass", "fence", "patio", "outdoor_furniture", "sky", "trees"]
    },
    "pool": {
        "name": "Pool",
        "icon": "room_pool",
        "keywords": ["pool", "swim", "spa", "hot_tub"],
        "features": ["pool_water", "deck", "lounge", "blue_water", "outdoor"]
    },
    "yard": {
        "name": "Yard",
        "icon": "room_yard",
        "keywords": ["yard", "garden", "landscape"],
        "features": ["grass", "trees", "plants", "fence", "sky"]
    },
    "garage": {
        "name": "Garage",
        "icon": "room_garage",
        "keywords": ["garage", "car", "parking"],
        "features": ["garage_door", "concrete", "storage", "car"]
    },
    "laundry": {
        "name": "Laundry Room",
        "icon": "room_laundry",
        "keywords": ["laundry", "washer", "dryer", "utility"],
        "features": ["washer", "dryer", "laundry_sink", "cabinets", "utility"]
    },
    "adu": {
        "name": "Guest House",
        "icon": "room_adu",
        "keywords": ["adu", "guest", "casita", "in-law", "accessory", "cottage", "unit"],
        "features": ["small_kitchen", "separate_entrance", "compact", "studio"]
    },
    "drone": {
        "name": "Drone Aerial",
        "icon": "room_drone",
        "keywords": ["drone", "aerial", "dji", "overhead"],
        "features": ["roof_visible", "birds_eye", "neighborhood", "sky_dominant"]
    },
    "detail": {
        "name": "Detail Shot",
        "icon": "room_detail",
        "keywords": ["detail", "close", "feature"],
        "features": ["close_up", "shallow_dof", "texture", "hardware"]
    },
    "sunroom": {
        "name": "Sunroom",
        "icon": "room_sunroom",
        "keywords": ["sunroom", "sun_room", "solarium", "conservatory", "florida_room"],
        "features": ["large_windows", "natural_light", "plants", "wicker", "bright"]
    },
    "side_yard": {
        "name": "Side Yard",
        "icon": "room_yard",
        "keywords": ["side_yard", "side", "narrow_yard", "passage"],
        "features": ["narrow", "fence", "gate", "pathway", "plants"]
    },
    "garden": {
        "name": "Garden",
        "icon": "room_garden",
        "keywords": ["garden", "flowers", "vegetable", "planter"],
        "features": ["plants", "flowers", "raised_beds", "soil", "greenery"]
    },
    "shed": {
        "name": "Shed",
        "icon": "room_shed",
        "keywords": ["shed", "storage", "outbuilding", "tool"],
        "features": ["small_structure", "storage", "tools", "outdoor"]
    },
    "driveway": {
        "name": "Driveway",
        "icon": "room_driveway",
        "keywords": ["driveway", "drive", "parking", "carport"],
        "features": ["pavement", "concrete", "asphalt", "cars", "garage_approach"]
    },
    "patio": {
        "name": "Patio",
        "icon": "room_patio",
        "keywords": ["patio", "deck", "terrace", "outdoor_living"],
        "features": ["outdoor_furniture", "pavers", "deck", "bbq", "umbrella"]
    }
}


# Standard home video structure order
# This is the typical flow for a real estate video:
# 1. Establishing shot (drone or exterior front)
# 2. Entry/transition into the home
# 3. Main living areas (living room, kitchen, dining)
# 4. Bedrooms and bathrooms
# 5. Office/extra rooms
# 6. Backyard and outdoor areas
# 7. Pull away/closing shot (drone)
HOME_VIDEO_ORDER = [
    # Opening - Establishing shots
    "drone",           # Aerial establishing shot
    "exterior_front",  # Front of house
    "driveway",        # Driveway
    "front_yard",      # Front yard
    "yard",            # General yard if no drone

    # Entry and main areas
    "entry",           # Entry/foyer
    "living_room",     # Main living space
    "sunroom",         # Sunroom (often off living area)
    "kitchen",         # Kitchen
    "dining",          # Dining room

    # Bedrooms
    "primary_bedroom", # Master/primary bedroom
    "bedroom",         # Other bedrooms

    # Bathrooms
    "bathroom",        # Bathrooms

    # Extra rooms
    "office",          # Office/study
    "laundry",         # Laundry room
    "garage",          # Garage
    "detail",          # Detail shots

    # ADU/Guest house (after main house, before outdoor)
    "adu",             # Accessory dwelling unit

    # Outdoor/closing
    "exterior_rear",   # Back of house
    "side_yard",       # Side yards
    "backyard",        # Backyard
    "patio",           # Patio/deck
    "garden",          # Garden
    "pool",            # Pool area
    "shed",            # Shed/outbuildings

    # Pull away is typically the last drone clip
]


# Standard photo order for real estate (from Aerial Canvas QA team)
# Naming: Photo-1, Photo-2, etc; Drone-1, Drone-2; Twilight-1, Twilight-2
PHOTO_ORDER = [
    "exterior_front",   # 1. Front of home / front patio
    "front_yard",       # (alternate for front exterior)
    "entry",            # 2. Entryway
    "living_room",      # 3. Living room
    "dining",           # 4. Dining room
    "kitchen",          # 5. Kitchen (note: team put family room before kitchen, but kitchen is 6)
    "primary_bedroom",  # 6. Master bedroom & bath
    "bedroom",          # 7. Secondary bedrooms
    "bathroom",         # (bathrooms can be grouped with bedrooms)
    "laundry",          # 8. Laundry area
    "office",           # (office can go with secondary rooms)
    "garage",           # (garage after main house)
    "exterior_rear",    # 9. Back of home
    "backyard",         # 9. Backyard
    "adu",              # 9. ADU
    "pool",             # 10. Community shots (pool, clubhouse, etc)
    "detail",           # Detail shots at end
    "drone",            # Drone photos separate category
]


def detect_photo_type(filename: str) -> str:
    """
    Detect if a photo is standard, drone, or twilight based on filename.
    Returns: 'photo', 'drone', or 'twilight'
    """
    fn_lower = filename.lower()
    if any(x in fn_lower for x in ['drone', 'aerial', 'dji', 'mavic', 'overhead']):
        return 'drone'
    elif any(x in fn_lower for x in ['twilight', 'dusk', 'sunset', 'evening', 'night', 'twi']):
        return 'twilight'
    else:
        return 'photo'


def sort_photos_for_delivery(photos: List[Dict]) -> List[Dict]:
    """
    Sort photos into the standard Aerial Canvas delivery order.

    Each photo dict should have:
    - filename: original filename
    - room_type: detected room type
    - path: file path (optional)

    Returns sorted list with added 'new_filename' and 'sort_order' fields.
    Naming format: 01-Front_Exterior.jpg, 02-Front_Exterior_2.jpg, etc.
    """
    if not photos:
        return photos

    # Separate by photo type
    standard_photos = [p for p in photos if detect_photo_type(p.get('filename', '')) == 'photo']
    drone_photos = [p for p in photos if detect_photo_type(p.get('filename', '')) == 'drone']
    twilight_photos = [p for p in photos if detect_photo_type(p.get('filename', '')) == 'twilight']

    # Create order map from PHOTO_ORDER
    order_map = {room: idx for idx, room in enumerate(PHOTO_ORDER)}

    def get_sort_key(photo):
        room = photo.get('room_type', 'unknown')
        # Unknown rooms go near the end
        return order_map.get(room, 50)

    # Sort each category by room order
    standard_photos.sort(key=get_sort_key)
    drone_photos.sort(key=get_sort_key)
    twilight_photos.sort(key=get_sort_key)

    def get_room_display_name(room_type: str) -> str:
        """Convert room_type key to display name for filename"""
        room_info = ROOM_TYPES.get(room_type, {})
        name = room_info.get('name', room_type)
        # Convert to filename-safe format: "Living Room" -> "Living_Room"
        return name.replace(' ', '_').replace('/', '_')

    def assign_descriptive_names(photo_list: list, prefix: str = None) -> list:
        """Assign numbered descriptive filenames, handling duplicates"""
        result = []
        room_counts = {}  # Track how many of each room type we've seen
        file_number = 1

        for photo in photo_list:
            photo = photo.copy()
            room_type = photo.get('room_type', 'unknown')
            room_name = get_room_display_name(room_type)

            # Track duplicates
            if room_type not in room_counts:
                room_counts[room_type] = 0
            room_counts[room_type] += 1

            # Build filename: 01-Living_Room or 02-Living_Room_2
            num_str = f"{file_number:02d}"
            if room_counts[room_type] == 1:
                # First of this room type - no suffix
                base_name = f"{num_str}-{room_name}"
            else:
                # Duplicate - add number suffix
                base_name = f"{num_str}-{room_name}_{room_counts[room_type]}"

            # Add prefix for drone/twilight
            if prefix:
                base_name = f"{num_str}-{prefix}_{room_name}"
                if room_counts[room_type] > 1:
                    base_name = f"{num_str}-{prefix}_{room_name}_{room_counts[room_type]}"

            photo['new_filename'] = base_name
            photo['sort_order'] = file_number
            photo['room_instance'] = room_counts[room_type]  # 1 = main, 2+ = alt
            result.append(photo)
            file_number += 1

        return result

    # Assign descriptive filenames to each category
    result = []

    # Standard photos first
    standard_result = assign_descriptive_names(standard_photos)
    for p in standard_result:
        p['photo_type'] = 'photo'
    result.extend(standard_result)

    # Drone photos
    drone_result = assign_descriptive_names(drone_photos, prefix="Drone")
    base_order = len(result)
    for i, p in enumerate(drone_result):
        p['photo_type'] = 'drone'
        p['sort_order'] = base_order + i + 1
        # Renumber with correct position
        num_str = f"{p['sort_order']:02d}"
        room_name = get_room_display_name(p.get('room_type', 'unknown'))
        if p.get('room_instance', 1) == 1:
            p['new_filename'] = f"{num_str}-Drone_{room_name}"
        else:
            p['new_filename'] = f"{num_str}-Drone_{room_name}_{p['room_instance']}"
    result.extend(drone_result)

    # Twilight photos
    twilight_result = assign_descriptive_names(twilight_photos, prefix="Twilight")
    base_order = len(result)
    for i, p in enumerate(twilight_result):
        p['photo_type'] = 'twilight'
        p['sort_order'] = base_order + i + 1
        # Renumber with correct position
        num_str = f"{p['sort_order']:02d}"
        room_name = get_room_display_name(p.get('room_type', 'unknown'))
        if p.get('room_instance', 1) == 1:
            p['new_filename'] = f"{num_str}-Twilight_{room_name}"
        else:
            p['new_filename'] = f"{num_str}-Twilight_{room_name}_{p['room_instance']}"
    result.extend(twilight_result)

    return result


def rename_photos_in_folder(folder_path: str, photos: List[Dict], preserve_extension: bool = True) -> Dict:
    """
    Actually rename photo files in a folder based on the sorted order.

    Returns dict with:
    - success: bool
    - renamed: list of {old, new} pairs
    - errors: list of error messages
    """
    results = {'success': True, 'renamed': [], 'errors': []}

    for photo in photos:
        old_path = photo.get('path') or os.path.join(folder_path, photo['filename'])
        if not os.path.exists(old_path):
            results['errors'].append(f"File not found: {photo['filename']}")
            continue

        # Get extension
        _, ext = os.path.splitext(photo['filename'])
        new_filename = f"{photo['new_filename']}{ext}" if preserve_extension else photo['new_filename']
        new_path = os.path.join(folder_path, new_filename)

        # Avoid overwriting existing files
        if os.path.exists(new_path) and new_path != old_path:
            # Add a temporary suffix
            temp_path = new_path + ".temp_rename"
            try:
                os.rename(old_path, temp_path)
                results['renamed'].append({'old': photo['filename'], 'new': new_filename, 'temp': True})
            except Exception as e:
                results['errors'].append(f"Error renaming {photo['filename']}: {e}")
                results['success'] = False
        else:
            try:
                os.rename(old_path, new_path)
                results['renamed'].append({'old': photo['filename'], 'new': new_filename})
            except Exception as e:
                results['errors'].append(f"Error renaming {photo['filename']}: {e}")
                results['success'] = False

    # Second pass: rename temp files to final names
    for item in results['renamed']:
        if item.get('temp'):
            temp_path = os.path.join(folder_path, item['new'] + ".temp_rename")
            final_path = os.path.join(folder_path, item['new'])
            try:
                os.rename(temp_path, final_path)
                item.pop('temp', None)
            except Exception as e:
                results['errors'].append(f"Error finalizing {item['new']}: {e}")

    return results


def sort_clips_for_timeline(clips: List[Dict], use_home_video_structure: bool = True) -> List[Dict]:
    """
    Sort clips into a logical timeline order based on home video structure.

    Returns clips ordered for a typical real estate video:
    1. Establishing shot (drone/exterior)
    2. Main rooms
    3. Backyard/outdoor
    4. Pull away (drone)
    """
    if not use_home_video_structure:
        return clips

    # Separate drone clips for opening/closing
    drone_clips = [c for c in clips if c['room_type'] == 'drone']
    non_drone_clips = [c for c in clips if c['room_type'] != 'drone']

    # Create order map
    order_map = {room: idx for idx, room in enumerate(HOME_VIDEO_ORDER)}

    # Sort non-drone clips by home video order
    def get_sort_key(clip):
        room = clip.get('room_type', 'unknown')
        # Put unknown rooms near the end but before outdoor
        return order_map.get(room, 50)

    sorted_clips = sorted(non_drone_clips, key=get_sort_key)

    # Structure the timeline:
    # - First drone clip as opener (if available)
    # - All sorted clips
    # - Last drone clip as pull away (if we have multiple drones)

    timeline = []

    # Opening drone shot
    if drone_clips:
        timeline.append(drone_clips[0])

    # Main content
    timeline.extend(sorted_clips)

    # Closing drone shot (if we have more than one)
    if len(drone_clips) > 1:
        timeline.append(drone_clips[-1])
    elif len(drone_clips) == 1 and not any(c['room_type'] in ['backyard', 'pool', 'exterior_rear'] for c in sorted_clips):
        # If only one drone clip and no outdoor shots, put it at the end as pull away
        # (it was already added as opener, so we'd duplicate - skip this case)
        pass

    return timeline


def extract_clip_frames(video_path: str, num_frames: int = 5) -> List[Tuple[float, np.ndarray]]:
    """
    Extract evenly-spaced frames from a video clip for analysis.
    Returns list of (timestamp, frame) tuples.
    """
    try:
        import cv2
    except ImportError:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps

    frames = []
    # Sample frames at even intervals, avoiding first/last 10%
    start = int(total_frames * 0.1)
    end = int(total_frames * 0.9)
    step = (end - start) // (num_frames + 1)

    for i in range(1, num_frames + 1):
        frame_num = start + (i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_num / fps
            frames.append((timestamp, frame))

    cap.release()
    return frames


def analyze_frame_for_room(frame) -> Dict:
    """
    Analyze a single frame to detect room characteristics.
    Returns dict with scores for each room type.
    """
    try:
        import cv2
    except ImportError:
        return {"unknown": 1.0}

    scores = {}
    h, w = frame.shape[:2]

    # Convert to different color spaces
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate basic features
    brightness = np.mean(gray)
    saturation = np.mean(hsv[:, :, 1])

    # Sky detection (top portion of frame, blue/bright)
    top_third = frame[:h//3, :, :]
    top_hsv = cv2.cvtColor(top_third, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(top_hsv, np.array([90, 20, 100]), np.array([140, 255, 255]))
    sky_ratio = np.sum(blue_mask > 0) / (blue_mask.shape[0] * blue_mask.shape[1])

    # Green detection (landscaping, grass, trees)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_ratio = np.sum(green_mask > 0) / (h * w)

    # Edge density (high = detailed interior, low = open exterior)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)

    # Tile/bathroom detection (regular patterns, high contrast lines)
    # Look for horizontal/vertical lines typical of tile
    lines_h = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    line_count = len(lines_h) if lines_h is not None else 0

    # Brown/warm wood detection (cabinets, floors)
    brown_mask = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([30, 255, 200]))
    wood_ratio = np.sum(brown_mask > 0) / (h * w)

    # White/bright surface detection (countertops, appliances)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    white_ratio = np.sum(white_mask > 0) / (h * w)

    # Scoring logic based on visual features

    # Drone/Aerial - lots of sky, roof visible, high angle
    if sky_ratio > 0.4:
        scores["drone"] = sky_ratio * 2

    # Exterior - sky + landscaping
    if sky_ratio > 0.2 and green_ratio > 0.1:
        scores["exterior_front"] = (sky_ratio + green_ratio) * 1.5
        scores["backyard"] = (sky_ratio + green_ratio) * 1.2
        scores["yard"] = (sky_ratio + green_ratio)

    # Kitchen - cabinets (brown) + white surfaces + high edge density
    if wood_ratio > 0.1 and white_ratio > 0.1 and edge_density > 0.1:
        scores["kitchen"] = (wood_ratio + white_ratio + edge_density) * 2

    # Bathroom - enhanced detection based on multiple bathroom-specific features
    bathroom_score = 0

    # 1. Tile detection - lots of regular lines from grout
    if line_count > 20:
        bathroom_score += 0.3
    elif line_count > 10:
        bathroom_score += 0.15

    # 2. High edge density from tile patterns
    if edge_density > 0.15:
        bathroom_score += 0.2
    elif edge_density > 0.1:
        bathroom_score += 0.1

    # 3. White/cream surfaces (toilets, tubs, sinks, vanities)
    if white_ratio > 0.25:
        bathroom_score += 0.25
    elif white_ratio > 0.15:
        bathroom_score += 0.15

    # 4. Blue-green tile colors (common in bathrooms)
    bluegreen_mask = cv2.inRange(hsv, np.array([80, 30, 80]), np.array([130, 255, 255]))
    bluegreen_ratio = np.sum(bluegreen_mask > 0) / (h * w)
    if bluegreen_ratio > 0.1:
        bathroom_score += 0.2
    elif bluegreen_ratio > 0.05:
        bathroom_score += 0.1

    # 5. Marble/veined pattern detection (common in luxury bathrooms)
    # High local contrast variance indicates veining
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    local_std = cv2.Laplacian(gray_blur, cv2.CV_64F).var()
    if local_std > 100 and white_ratio > 0.2:
        bathroom_score += 0.15

    # 6. High reflectivity (mirrors, glass, chrome)
    # Very bright spots in the image
    very_bright = cv2.inRange(hsv, np.array([0, 0, 240]), np.array([180, 20, 255]))
    bright_ratio = np.sum(very_bright > 0) / (h * w)
    if bright_ratio > 0.05:
        bathroom_score += 0.1

    # 7. Pink/coral tile (seen in some bathrooms)
    pink_mask = cv2.inRange(hsv, np.array([0, 30, 150]), np.array([15, 150, 255]))
    pink_ratio = np.sum(pink_mask > 0) / (h * w)
    if pink_ratio > 0.1:
        bathroom_score += 0.15

    # Assign bathroom score if significant features detected
    if bathroom_score > 0.3:
        scores["bathroom"] = bathroom_score * 2.5  # Boost bathroom detection

    # Bedroom - moderate features, lower edge density, warmer tones
    if edge_density < 0.1 and wood_ratio > 0.05:
        scores["bedroom"] = 0.5 + (1 - edge_density)
        scores["primary_bedroom"] = 0.4 + (1 - edge_density)

    # Living room - enhanced detection based on training images
    # Key characteristics: open space, good natural light, warm colors, area rugs
    living_score = 0.0

    # 1. Open space with lower edge density (not cluttered like kitchen)
    if edge_density < 0.12:
        living_score += 0.3

    # 2. Good brightness (well-lit spaces)
    if brightness > 100:
        living_score += 0.2

    # 3. Warm tones (beige, brown, cream - common in living rooms)
    warm_mask = cv2.inRange(hsv, np.array([10, 20, 100]), np.array([30, 150, 255]))
    warm_ratio = np.sum(warm_mask > 0) / (h * w)
    if warm_ratio > 0.1:
        living_score += 0.2

    # 4. Wood tones (hardwood floors, furniture)
    if wood_ratio > 0.05:
        living_score += 0.15

    # 5. NOT a bathroom (low white tile, no strong blue/green tones)
    if white_ratio < 0.4 and bathroom_score < 0.3:
        living_score += 0.15

    # 6. Larger aspect ratio areas (living rooms tend to be wide shots)
    if w > h * 1.2:  # Wide shot
        living_score += 0.1

    if living_score > 0.3:
        scores["living_room"] = living_score

    # Entry - narrower, hallway-like proportions
    if edge_density > 0.08 and edge_density < 0.15:
        scores["entry"] = 0.5

    # Office - moderate everything
    scores["office"] = 0.3

    # Garage - concrete colors, large door patterns
    gray_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 30, 180]))
    gray_ratio = np.sum(gray_mask > 0) / (h * w)
    if gray_ratio > 0.3 and edge_density < 0.1:
        scores["garage"] = gray_ratio

    # Detail shots - shallow DOF indicator (variable sharpness)
    # Check if center is sharper than edges
    center_region = gray[h//3:2*h//3, w//3:2*w//3]
    edge_region_top = gray[:h//4, :]
    center_laplacian = cv2.Laplacian(center_region, cv2.CV_64F).var()
    edge_laplacian = cv2.Laplacian(edge_region_top, cv2.CV_64F).var()
    if center_laplacian > edge_laplacian * 2:
        scores["detail"] = (center_laplacian / (edge_laplacian + 1)) * 0.3

    # Normalize scores
    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    return scores


def load_reference_images() -> Dict[str, List]:
    """
    Load reference images from room-type folders for comparison-based classification.
    Looks for folders like 'Bathrooms/', 'Kitchens/', etc. in the app directory.
    """
    import cv2
    references = {}
    app_dir = os.path.dirname(__file__)

    # Map folder names to room type keys
    folder_mapping = {
        "Bathrooms": "bathroom",
        "Kitchens": "kitchen",
        "LivingRooms": "living_room",
        "Bedrooms": "bedroom",
        "Exteriors": "exterior_front",
    }

    for folder_name, room_key in folder_mapping.items():
        folder_path = os.path.join(app_dir, folder_name)
        if os.path.isdir(folder_path):
            histograms = []
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

            for fname in os.listdir(folder_path):
                ext = os.path.splitext(fname)[1].lower()
                if ext in image_extensions:
                    img_path = os.path.join(folder_path, fname)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize for consistent comparison
                            img = cv2.resize(img, (200, 200))
                            # Convert to HSV for better color matching
                            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                            # Calculate color histogram
                            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                            histograms.append(hist)
                    except Exception:
                        pass

            if histograms:
                references[room_key] = histograms

    return references


def compare_to_references(frame, references: Dict) -> Dict[str, float]:
    """
    Compare a frame against reference images using histogram correlation.
    Returns scores for each room type with references.
    """
    import cv2
    scores = {}

    # Prepare frame histogram
    frame_resized = cv2.resize(frame, (200, 200))
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    frame_hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(frame_hist, frame_hist, 0, 1, cv2.NORM_MINMAX)

    for room_key, ref_hists in references.items():
        # Compare against all reference images, take best match
        best_match = 0
        for ref_hist in ref_hists:
            correlation = cv2.compareHist(frame_hist, ref_hist, cv2.HISTCMP_CORREL)
            best_match = max(best_match, correlation)

        if best_match > 0.3:  # Only count if reasonably similar
            scores[room_key] = best_match

    return scores


# Load reference images on startup (cached)
_reference_images_cache = None

def get_reference_images():
    global _reference_images_cache
    if _reference_images_cache is None:
        _reference_images_cache = load_reference_images()
    return _reference_images_cache


def classify_clip_room(video_path: str, filename: str = "") -> Tuple[str, float]:
    """
    Classify what room type a clip shows.
    Returns (room_type_key, confidence).
    Uses YOLO object detection, visual analysis, reference image comparison,
    and photo room detections from the same folder (cross-training).
    """
    # First check filename for keywords
    name_lower = filename.lower() if filename else os.path.basename(video_path).lower()

    for room_key, room_info in ROOM_TYPES.items():
        for keyword in room_info["keywords"]:
            if keyword in name_lower:
                return (room_key, 0.95)  # High confidence if filename matches

    # Analyze frames visually
    frames = extract_clip_frames(video_path, num_frames=5)
    if not frames:
        return ("unknown", 0.0)

    # Load reference images
    references = get_reference_images()

    # Get photo room hints from same folder (cross-training)
    folder_path = os.path.dirname(video_path)
    photo_hints = {}
    try:
        hints = stats_tracker.get_photo_room_hints(folder_path)
        # Convert hints to scores - photos from same folder give hints about what rooms exist
        for room, detections in hints.items():
            # Higher confidence photo detections = stronger hint
            avg_confidence = sum(d[1] for d in detections) / len(detections) if detections else 0
            if avg_confidence > 0.4:
                photo_hints[room] = avg_confidence * 0.3  # 30% weight boost from photo hints
    except Exception:
        pass  # Photo hints are optional

    # Aggregate scores from all frames
    total_scores = {}
    reference_scores = {}
    yolo_scores = {}  # YOLO object detection scores

    for timestamp, frame in frames:
        # 1. YOLO Object Detection (highest priority - actually sees objects)
        if YOLO_AVAILABLE:
            detected_objects = detect_objects_in_frame(frame)
            if detected_objects:
                object_room_scores = score_room_from_objects(detected_objects)
                for room, score in object_room_scores.items():
                    yolo_scores[room] = yolo_scores.get(room, 0) + score

        # 2. Visual analysis scores (color, edges, patterns)
        frame_scores = analyze_frame_for_room(frame)
        for room, score in frame_scores.items():
            total_scores[room] = total_scores.get(room, 0) + score

        # 3. Reference image comparison scores (if references exist)
        if references:
            ref_scores = compare_to_references(frame, references)
            for room, score in ref_scores.items():
                reference_scores[room] = reference_scores.get(room, 0) + score

    # Average all scores
    num_frames = len(frames)

    if total_scores:
        for room in total_scores:
            total_scores[room] /= num_frames

    if reference_scores:
        for room in reference_scores:
            reference_scores[room] /= num_frames

    if yolo_scores:
        for room in yolo_scores:
            yolo_scores[room] /= num_frames

    # Combine scores with weighted priorities
    # YOLO is most reliable (actually sees objects), then references, then visual heuristics
    # Photo hints from same folder provide cross-training boost
    combined_scores = {}

    # Start with visual heuristic scores (baseline)
    for room, score in total_scores.items():
        combined_scores[room] = score * 0.3  # 30% weight

    # Add reference image scores
    for room, score in reference_scores.items():
        if score > 0.3:
            combined_scores[room] = combined_scores.get(room, 0) + (score * 0.4)  # 40% weight

    # Add YOLO scores (highest weight - these are actual object detections)
    for room, score in yolo_scores.items():
        combined_scores[room] = combined_scores.get(room, 0) + (score * 1.5)  # 150% weight (dominant)

    # Add photo hints from same folder (cross-training from photos)
    # This helps when photos from the same property were already analyzed
    for room, hint_score in photo_hints.items():
        if room in combined_scores:
            # Boost existing room scores if photos also detected this room
            combined_scores[room] = combined_scores[room] * (1 + hint_score)
        else:
            # Add room as possibility if only photos detected it
            combined_scores[room] = hint_score * 0.5

    # Find best match
    if combined_scores:
        best_room = max(combined_scores, key=combined_scores.get)
        # Normalize confidence to 0-1 range
        max_score = combined_scores[best_room]
        confidence = min(max_score, 1.0) if max_score <= 1.0 else min(max_score / 2, 1.0)
        return (best_room, confidence)

    return ("unknown", 0.0)


def classify_photo_room(image_path: str, filename: str = "", img=None, image_bytes: bytes = None) -> Tuple[str, float]:
    """
    Classify what room type a photo shows.
    Returns (room_type_key, confidence).
    Uses learning database first, then YOLO object detection and visual analysis.
    Can optionally accept pre-loaded image or image_bytes to avoid re-reading.
    """
    import cv2

    # FIRST: Check learning database for this exact image
    if image_bytes:
        try:
            learned = learning_db.get_learned_room(image_bytes)
            if learned:
                room_type, confidence = learned
                return (room_type, confidence)  # Return learned classification
        except Exception:
            pass  # Continue with normal classification if learning db fails

    # Check filename for keywords
    name_lower = filename.lower() if filename else os.path.basename(image_path).lower()

    for room_key, room_info in ROOM_TYPES.items():
        for keyword in room_info["keywords"]:
            if keyword in name_lower:
                return (room_key, 0.95)  # High confidence if filename matches

    # Load image if not provided
    if img is None:
        img = cv2.imread(image_path)
        if img is None:
            return ("unknown", 0.0)

    # Downscale for faster analysis
    max_size = 800
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Score using same methods as video
    total_scores = {}
    yolo_scores = {}

    # 1. YOLO Object Detection
    if YOLO_AVAILABLE:
        detected_objects = detect_objects_in_frame(img)
        if detected_objects:
            object_room_scores = score_room_from_objects(detected_objects)
            for room, score in object_room_scores.items():
                yolo_scores[room] = score

    # 2. Visual analysis scores (color, edges, patterns)
    frame_scores = analyze_frame_for_room(img)
    for room, score in frame_scores.items():
        total_scores[room] = score

    # 3. Reference image comparison
    references = get_reference_images()
    reference_scores = {}
    if references:
        ref_scores = compare_to_references(img, references)
        for room, score in ref_scores.items():
            reference_scores[room] = score

    # Combine scores with weighted priorities
    combined_scores = {}

    # Visual heuristic scores (baseline)
    for room, score in total_scores.items():
        combined_scores[room] = score * 0.3

    # Reference image scores
    for room, score in reference_scores.items():
        if score > 0.3:
            combined_scores[room] = combined_scores.get(room, 0) + (score * 0.4)

    # YOLO scores (highest weight)
    for room, score in yolo_scores.items():
        combined_scores[room] = combined_scores.get(room, 0) + (score * 1.5)

    # Find best match
    if combined_scores:
        best_room = max(combined_scores, key=combined_scores.get)
        max_score = combined_scores[best_room]
        confidence = min(max_score, 1.0) if max_score <= 1.0 else min(max_score / 2, 1.0)
        return (best_room, confidence)

    return ("unknown", 0.0)


def extract_frame_at_timestamp(video_path: str, timestamp: float, width: int = 160) -> Optional[str]:
    """
    Extract a single frame from a video at a specific timestamp.
    Returns base64-encoded JPEG for inline HTML display.
    """
    try:
        import cv2
        import base64
    except ImportError:
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_num = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        # Resize to thumbnail
        h, w = frame.shape[:2]
        aspect = w / h
        new_w = width
        new_h = int(new_w / aspect)
        frame = cv2.resize(frame, (new_w, new_h))

        # Encode to JPEG base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    except Exception:
        return None


def analyze_clip_quality_quick(video_path: str) -> Dict:
    """
    Quick clip analysis - just gets duration and basic info.
    Uses simple heuristic: trim 15% from start and end (typical setup/teardown).
    Much faster than full optical flow analysis.
    """
    try:
        import cv2
    except ImportError:
        return {"error": "OpenCV not available"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    # Simple heuristic: trim 15% from each end
    trim_pct = 0.15
    hero_in = duration * trim_pct
    hero_out = duration * (1 - trim_pct)
    hero_duration = hero_out - hero_in

    return {
        "duration": duration,
        "fps": fps,
        "total_frames": total_frames,
        "avg_stability": 0.75,  # Assume decent
        "avg_exposure": 0.7,
        "best_in_point": hero_in,
        "best_out_point": hero_out,
        "best_segment_score": 0.7,
        "shake_start": hero_in,
        "shake_end": duration - hero_out,
        "hero_duration": hero_duration,
        "trim_suggestion": f"Quick trim: {hero_in:.1f}s start, {duration - hero_out:.1f}s end (15% each)"
    }


def analyze_clip_quality(video_path: str, quick_mode: bool = False) -> Dict:
    """
    Smart clip analysis for gimbal footage workflow:

    Gimbal shooting pattern:
    1. Press record → setup shake (unusable)
    2. Operator settles → smooth controlled movement (THE HERO SHOT)
    3. End of move → teardown shake (unusable)

    This function detects the transition from shaky→smooth and smooth→shaky
    to automatically find the usable "hero" portion of each clip.

    Args:
        quick_mode: If True, use fast heuristic instead of optical flow

    Uses (full mode):
    - Optical flow for precise motion analysis
    - Motion consistency (smooth gimbal vs erratic handheld)
    - Acceleration detection (sudden stops/starts)
    - Real estate video timing knowledge
    """
    if quick_mode:
        return analyze_clip_quality_quick(video_path)
    try:
        import cv2
    except ImportError:
        return {"error": "OpenCV not available"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # For very short clips, skip heavy analysis
    if duration < 2:
        cap.release()
        return {
            "duration": duration,
            "fps": fps,
            "total_frames": total_frames,
            "avg_stability": 0.8,
            "avg_exposure": 0.7,
            "best_in_point": 0,
            "best_out_point": duration,
            "best_segment_score": 0.7,
            "shake_start": 0,
            "shake_end": 0,
            "hero_duration": duration,
            "trim_suggestion": "Clip too short for analysis"
        }

    # Sample rate based on fps (analyze ~6 frames per second)
    # This catches quick shakes while being efficient
    sample_interval = max(int(fps / 6), 1)

    # Motion metrics per sample
    timestamps = []
    motion_magnitudes = []      # Overall motion amount
    motion_consistency = []     # How uniform is the motion direction
    exposure_scores = []
    motion_accelerations = []   # Change in motion (jerk detection)

    prev_gray = None
    prev_motion = 0
    frame_idx = 0

    # Optical flow parameters (tuned for gimbal detection)
    flow_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Resize for faster processing
            scale = 0.25
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            timestamp = frame_idx / fps
            timestamps.append(timestamp)

            # Exposure analysis
            brightness = np.mean(gray) / 255
            # Ideal is 0.4-0.6 range (well exposed interior)
            if brightness < 0.15:
                exposure_score = brightness / 0.15 * 0.5  # Dark penalty
            elif brightness > 0.85:
                exposure_score = (1 - brightness) / 0.15 * 0.5  # Blown out penalty
            else:
                exposure_score = 1 - abs(brightness - 0.5) * 1.5
            exposure_scores.append(max(0, min(1, exposure_score)))

            if prev_gray is not None:
                # Optical flow analysis
                try:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)

                    # Motion magnitude (how much movement)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    avg_magnitude = np.mean(mag)
                    motion_magnitudes.append(avg_magnitude)

                    # Motion consistency (gimbal = consistent direction, shake = random)
                    # Calculate the standard deviation of motion angles
                    # Low std = consistent direction (good gimbal move)
                    # High std = erratic motion (shake/setup)

                    # Only consider pixels with significant motion
                    motion_mask = mag > np.percentile(mag, 50)
                    if np.sum(motion_mask) > 100:
                        angles_moving = ang[motion_mask]
                        # Circular standard deviation for angles
                        sin_sum = np.mean(np.sin(angles_moving))
                        cos_sum = np.mean(np.cos(angles_moving))
                        r = np.sqrt(sin_sum**2 + cos_sum**2)
                        consistency = r  # 0 = random directions, 1 = all same direction
                    else:
                        consistency = 1.0  # No motion = consistent
                    motion_consistency.append(consistency)

                    # Acceleration (jerk detection)
                    # Smooth gimbal moves have gradual acceleration
                    # Setup/teardown has sudden jerks
                    acceleration = abs(avg_magnitude - prev_motion)
                    motion_accelerations.append(acceleration)
                    prev_motion = avg_magnitude

                except Exception:
                    motion_magnitudes.append(0)
                    motion_consistency.append(1.0)
                    motion_accelerations.append(0)
            else:
                motion_magnitudes.append(0)
                motion_consistency.append(1.0)
                motion_accelerations.append(0)

            prev_gray = gray

        frame_idx += 1

    cap.release()

    if len(timestamps) < 5:
        return {"error": "Not enough frames analyzed"}

    # =========================================================================
    # SMART TRIM: Find the hero section
    # =========================================================================

    # Smooth the signals to reduce noise (rolling average)
    def smooth(data, window=5):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='same')

    motion_smooth = smooth(motion_magnitudes, 5)
    accel_smooth = smooth(motion_accelerations, 5)
    consistency_smooth = smooth(motion_consistency, 3)

    # Calculate a "usability score" for each sample point
    # High score = smooth, consistent gimbal movement
    # Low score = shaky, erratic setup/teardown

    usability_scores = []
    for i in range(len(timestamps)):
        # Factors that make footage usable:
        # 1. Low acceleration (no sudden jerks) - most important for gimbal
        # 2. High consistency (motion in one direction)
        # 3. Moderate motion (not static, not too fast)
        # 4. Good exposure

        motion = motion_smooth[i] if i < len(motion_smooth) else 0
        accel = accel_smooth[i] if i < len(accel_smooth) else 0
        consist = consistency_smooth[i] if i < len(consistency_smooth) else 1
        exposure = exposure_scores[i] if i < len(exposure_scores) else 0.5

        # Normalize motion (ideal is 0.5-2.0 for gimbal moves)
        motion_score = 1.0
        if motion < 0.3:
            motion_score = 0.7  # Too static (might be setup pause)
        elif motion > 4.0:
            motion_score = max(0, 1 - (motion - 4) / 4)  # Too fast (shake)

        # Acceleration penalty (shake = high acceleration variance)
        accel_score = max(0, 1 - accel * 2)

        # Combine scores
        # Acceleration is most important for detecting shake
        usability = (
            accel_score * 0.45 +      # Jerk/shake detection
            consist * 0.30 +           # Direction consistency
            motion_score * 0.15 +      # Motion amount
            exposure * 0.10            # Exposure quality
        )
        usability_scores.append(usability)

    usability_smooth = smooth(usability_scores, 7)

    # Find the "hero section" - longest continuous high-usability segment
    # Use a threshold approach: find where usability crosses above/below threshold

    # Adaptive threshold based on clip's best sections
    threshold = np.percentile(usability_smooth, 60)  # Top 40% is "usable"
    threshold = max(threshold, 0.5)  # But at least 0.5

    # Find segments above threshold
    above_threshold = usability_smooth >= threshold

    # Find the longest continuous segment
    segments = []
    start_idx = None
    for i, above in enumerate(above_threshold):
        if above and start_idx is None:
            start_idx = i
        elif not above and start_idx is not None:
            segments.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(above_threshold) - 1))

    # Pick the best segment (longest, with good average score)
    best_segment = None
    best_segment_score = 0

    for start, end in segments:
        length = end - start
        if length < 3:  # Skip very short segments
            continue
        avg_score = np.mean(usability_smooth[start:end+1])
        # Prefer longer segments with good scores
        segment_value = length * avg_score
        if segment_value > best_segment_score:
            best_segment_score = segment_value
            best_segment = (start, end)

    # If no good segment found, use middle 60% of clip
    if best_segment is None:
        start_idx = int(len(timestamps) * 0.2)
        end_idx = int(len(timestamps) * 0.8)
        best_segment = (start_idx, end_idx)

    # Convert to timestamps
    hero_in = timestamps[best_segment[0]]
    hero_out = timestamps[best_segment[1]]

    # Add small buffer for editing (0.5 seconds on each end if possible)
    buffer = 0.5
    hero_in = max(0, hero_in - buffer)
    hero_out = min(duration, hero_out + buffer)

    # Calculate how much we're trimming
    shake_start_duration = hero_in
    shake_end_duration = duration - hero_out
    hero_duration = hero_out - hero_in

    # Generate trim suggestion text
    if shake_start_duration > 1.0 or shake_end_duration > 1.0:
        trim_suggestion = f"Trim {shake_start_duration:.1f}s from start, {shake_end_duration:.1f}s from end"
    elif shake_start_duration > 0.3 or shake_end_duration > 0.3:
        trim_suggestion = f"Minor trim: {shake_start_duration:.1f}s start, {shake_end_duration:.1f}s end"
    else:
        trim_suggestion = "Clip is clean - minimal trim needed"

    # Calculate overall stability (for the hero section)
    hero_start_idx = best_segment[0]
    hero_end_idx = best_segment[1]
    hero_stability = np.mean(usability_smooth[hero_start_idx:hero_end_idx+1])

    return {
        "duration": duration,
        "fps": fps,
        "total_frames": total_frames,
        "avg_stability": hero_stability,
        "avg_exposure": np.mean(exposure_scores[hero_start_idx:hero_end_idx+1]) if hero_end_idx > hero_start_idx else np.mean(exposure_scores),
        "best_in_point": hero_in,
        "best_out_point": hero_out,
        "best_segment_score": hero_stability,
        "shake_start": shake_start_duration,
        "shake_end": shake_end_duration,
        "hero_duration": hero_duration,
        "trim_suggestion": trim_suggestion,
        "usability_timeline": list(zip(timestamps, usability_smooth.tolist() if hasattr(usability_smooth, 'tolist') else list(usability_smooth))),
        "stability_timeline": list(zip(timestamps, usability_scores)),
        "exposure_timeline": list(zip(timestamps, exposure_scores))
    }


def get_descriptive_clip_name(clip_index: int, room_type: str, room_instance: int = 1) -> str:
    """
    Generate a descriptive clip name like 01-Living_Room or 02-Kitchen_2
    """
    room_info = ROOM_TYPES.get(room_type, {})
    room_name = room_info.get('name', room_type)
    # Convert to filename-safe format: "Living Room" -> "Living_Room"
    room_name_safe = room_name.replace(' ', '_').replace('/', '_')

    num_str = f"{clip_index:02d}"
    if room_instance == 1:
        return f"{num_str}-{room_name_safe}"
    else:
        return f"{num_str}-{room_name_safe}_{room_instance}"


def assign_descriptive_clip_names(clips: List[Dict]) -> List[Dict]:
    """
    Assign descriptive names to video clips, handling duplicates.
    Returns clips with added 'display_name' and 'clip_instance' fields.
    """
    room_counts = {}  # Track how many of each room type we've seen
    result = []

    for idx, clip in enumerate(clips):
        clip = clip.copy()
        room_type = clip.get('room_type', 'unknown')

        # Track duplicates
        if room_type not in room_counts:
            room_counts[room_type] = 0
        room_counts[room_type] += 1

        clip['clip_instance'] = room_counts[room_type]
        clip['display_name'] = get_descriptive_clip_name(idx + 1, room_type, room_counts[room_type])
        clip['sort_order'] = idx + 1
        result.append(clip)

    return result


def generate_fcpxml(clips: List[Dict], project_name: str = "Auto_Sort_Timeline") -> str:
    """
    Generate Final Cut Pro X XML (.fcpxml) for the organized clips.
    Each dict in clips should have: path, room_type, in_point, out_point, duration
    Uses descriptive naming like 01-Living_Room, 02-Kitchen, etc.
    """
    from urllib.parse import quote
    import html

    # Assign descriptive names to clips
    clips = assign_descriptive_clip_names(clips)

    # Helper to create proper file URL
    def file_url(path: str) -> str:
        # Ensure absolute path and proper URL encoding
        abs_path = os.path.abspath(path)
        # URL encode the path (but not the slashes)
        encoded_path = quote(abs_path, safe='/')
        return f"file://{encoded_path}"

    # Helper to escape XML special characters
    def xml_escape(text: str) -> str:
        return html.escape(str(text))

    # Detect frame rate from first clip (default to 30fps)
    fps = 30
    width = 1920
    height = 1080

    if clips:
        first_clip = clips[0]
        # Try to get actual specs from metadata
        if 'metadata' in first_clip:
            meta = first_clip['metadata']
            fps = meta.get('fps', 30)
            width = meta.get('width', 1920)
            height = meta.get('height', 1080)

    # Determine format name based on resolution
    if height >= 2160:
        format_name = f"FFVideoFormat2160p{int(fps)}"
    elif height >= 1080:
        format_name = f"FFVideoFormat1080p{int(fps)}"
    else:
        format_name = f"FFVideoFormat720p{int(fps)}"

    # Build FCPXML structure (version 1.8 for broader compatibility)
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE fcpxml>',
        '<fcpxml version="1.8">',
        '  <resources>',
    ]

    # Add format resource
    frame_duration = f"100/{int(fps * 100)}s"  # e.g., "100/3000s" for 30fps
    xml_lines.append(f'    <format id="r1" name="{format_name}" frameDuration="{frame_duration}" width="{width}" height="{height}"/>')

    # Add asset resources for each clip
    for idx, clip in enumerate(clips):
        clip_id = f"r{idx + 10}"
        duration_secs = clip.get("duration", 10)
        duration_rational = f"{int(duration_secs * fps * 100)}/{int(fps * 100)}s"
        # Use descriptive name like "01-Living_Room" instead of original filename
        clip_name = xml_escape(clip.get("display_name", os.path.basename(clip.get("path", f"clip_{idx}"))))
        clip_url = file_url(clip.get("path", ""))

        xml_lines.append(
            f'    <asset id="{clip_id}" name="{clip_name}" uid="{clip_id}" '
            f'src="{clip_url}" start="0s" duration="{duration_rational}" '
            f'hasVideo="1" hasAudio="1" format="r1"/>'
        )

    xml_lines.append('  </resources>')
    xml_lines.append('  <library>')
    xml_lines.append(f'    <event name="{xml_escape(project_name)}">')
    xml_lines.append(f'      <project name="{xml_escape(project_name)}">')
    xml_lines.append(f'        <sequence format="r1" tcStart="0s" tcFormat="NDF">')
    xml_lines.append('          <spine>')

    # Add clips to timeline in order (already sorted)
    timeline_position = 0
    for idx, clip in enumerate(clips):
        clip_id = f"r{idx + 10}"

        in_point = clip.get("in_point", 0)
        out_point = clip.get("out_point", clip.get("duration", 10))
        clip_duration = out_point - in_point

        # Convert to rational time format
        offset_rational = f"{int(timeline_position * fps * 100)}/{int(fps * 100)}s"
        duration_rational = f"{int(clip_duration * fps * 100)}/{int(fps * 100)}s"
        start_rational = f"{int(in_point * fps * 100)}/{int(fps * 100)}s"

        # Use descriptive name like "01-Living_Room"
        clip_name = xml_escape(clip.get("display_name", os.path.basename(clip.get("path", f"clip_{idx}"))))
        room_type = clip.get("room_type", "unknown")
        room_name = ROOM_TYPES.get(room_type, {}).get("name", room_type)

        # Add comment for room type (original filename for reference)
        original_file = os.path.basename(clip.get("path", ""))
        xml_lines.append(f'          <!-- {clip_name}: {room_name} (from {original_file}) -->')
        xml_lines.append(
            f'          <asset-clip ref="{clip_id}" offset="{offset_rational}" '
            f'name="{clip_name}" duration="{duration_rational}" start="{start_rational}"/>'
        )

        timeline_position += clip_duration

    xml_lines.extend([
        '          </spine>',
        '        </sequence>',
        '      </project>',
        '    </event>',
        '  </library>',
        '</fcpxml>'
    ])

    return '\n'.join(xml_lines)


def generate_premiere_xml(clips: List[Dict], project_name: str = "Auto_Sort_Timeline") -> str:
    """
    Generate Adobe Premiere Pro XML (.xml) for the organized clips.
    Uses descriptive naming like 01-Living_Room, 02-Kitchen, etc.
    """
    from urllib.parse import quote
    import html

    # Assign descriptive names to clips
    clips = assign_descriptive_clip_names(clips)

    # Helper to create proper file URL for Premiere
    def file_url(path: str) -> str:
        abs_path = os.path.abspath(path)
        encoded_path = quote(abs_path, safe='/')
        return f"file://localhost{encoded_path}"

    # Helper to escape XML special characters
    def xml_escape(text: str) -> str:
        return html.escape(str(text))

    fps = 30
    timebase = 30

    # Try to detect fps from first clip
    if clips and 'metadata' in clips[0]:
        fps = clips[0]['metadata'].get('fps', 30)
        timebase = int(fps)

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="4">',
        '  <sequence>',
        f'    <name>{xml_escape(project_name)}</name>',
        '    <rate>',
        f'      <timebase>{timebase}</timebase>',
        '      <ntsc>FALSE</ntsc>',
        '    </rate>',
        '    <media>',
        '      <video>',
        '        <track>',
    ]

    # Process clips in order (already sorted)
    timeline_position = 0
    for idx, clip in enumerate(clips):
        in_point = clip.get("in_point", 0)
        out_point = clip.get("out_point", clip.get("duration", 10))
        total_duration = clip.get("duration", 10)

        in_frames = int(in_point * fps)
        out_frames = int(out_point * fps)
        duration_frames = out_frames - in_frames
        total_frames = int(total_duration * fps)

        # Use descriptive name like "01-Living_Room"
        clip_name = xml_escape(clip.get("display_name", os.path.basename(clip.get("path", f"clip_{idx}"))))
        clip_url = file_url(clip.get("path", ""))

        xml_lines.extend([
            '          <clipitem>',
            f'            <name>{clip_name}</name>',
            f'            <duration>{total_frames}</duration>',
            '            <rate>',
            f'              <timebase>{timebase}</timebase>',
            '            </rate>',
            f'            <start>{timeline_position}</start>',
            f'            <end>{timeline_position + duration_frames}</end>',
            f'            <in>{in_frames}</in>',
            f'            <out>{out_frames}</out>',
            '            <file>',
            f'              <pathurl>{clip_url}</pathurl>',
            '            </file>',
            '          </clipitem>',
        ])
        timeline_position += duration_frames

    xml_lines.extend([
        '        </track>',
        '      </video>',
        '    </media>',
        '  </sequence>',
        '</xmeml>'
    ])

    return '\n'.join(xml_lines)


def generate_resolve_xml(clips: List[Dict], project_name: str = "Auto_Sort_Timeline") -> str:
    """
    Generate DaVinci Resolve compatible XML (uses same format as Premiere).
    """
    # DaVinci Resolve can import Premiere XML, so we use the same format
    return generate_premiere_xml(clips, project_name)


def scan_folder_for_clips(folder_path: str) -> List[Dict]:
    """
    Scan a folder for video clips and analyze them.
    Returns list of clip info dicts.
    """
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.m4v'}
    clips = []

    if not os.path.isdir(folder_path):
        return clips

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in video_extensions:
                filepath = os.path.join(root, filename)

                # Classify room type
                room_type, confidence = classify_clip_room(filepath, filename)

                # Analyze quality for best moments
                quality = analyze_clip_quality(filepath)

                clips.append({
                    "path": filepath,
                    "filename": filename,
                    "room_type": room_type,
                    "room_confidence": confidence,
                    "duration": quality.get("duration", 0),
                    "fps": quality.get("fps", 30),
                    "in_point": quality.get("best_in_point", 0),
                    "out_point": quality.get("best_out_point", quality.get("duration", 10)),
                    "stability_score": quality.get("avg_stability", 0),
                    "exposure_score": quality.get("avg_exposure", 0),
                    "segment_score": quality.get("best_segment_score", 0),
                    "shake_start": quality.get("shake_start", 0),
                    "shake_end": quality.get("shake_end", 0),
                    "hero_duration": quality.get("hero_duration", quality.get("duration", 0)),
                    "trim_suggestion": quality.get("trim_suggestion", "")
                })

    return clips


def display_auto_sort():
    """
    Auto Sort feature - analyzes raw footage and generates edit-ready XML:
    1. Connects to Dropbox folder with raw clips
    2. Analyzes each clip to identify room type (bedroom, bathroom, kitchen, etc.)
    3. Finds the best moments in each clip (stability, exposure, composition)
    4. Generates XML timeline for DaVinci Resolve, Premiere Pro, or Final Cut Pro X
    """

    # Auto Sort icon (folder with magic wand)
    auto_sort_icon = '''<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#7B8CDE" stroke-width="2">
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
        <path d="M12 11v6M9 14l3-3 3 3" stroke="#7B8CDE" stroke-width="2"></path>
    </svg>'''

    # Initialize sort mode in session state
    if 'auto_sort_mode' not in st.session_state:
        st.session_state.auto_sort_mode = 'Video'

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 8px;">
            {auto_sort_icon}
            <h2 style="color: #fff; margin: 0;">Auto Sort</h2>
        </div>
        <p style="color: #a1a1aa; font-size: 14px; margin-bottom: 16px;">Organize & rename files by room type for delivery</p>
    </div>
    """, unsafe_allow_html=True)

    # Centered mode selector buttons
    _, btn_col1, btn_col2, _ = st.columns([1.5, 1, 1, 1.5])

    with btn_col1:
        video_selected = st.session_state.auto_sort_mode == 'Video'
        if st.button(
            "Video",
            key="sort_mode_video",
            type="primary" if video_selected else "secondary",
            use_container_width=True
        ):
            st.session_state.auto_sort_mode = 'Video'
            st.rerun()

    with btn_col2:
        photos_selected = st.session_state.auto_sort_mode == 'Photos'
        if st.button(
            "Photos",
            key="sort_mode_photos",
            type="primary" if photos_selected else "secondary",
            use_container_width=True
        ):
            st.session_state.auto_sort_mode = 'Photos'
            st.rerun()

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # Get current mode
    sort_mode = st.session_state.auto_sort_mode

    # =============================================
    # PHOTO SORTING MODE
    # =============================================
    if sort_mode == "Photos":
        # Feature overview cards for PHOTOS
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px;">
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('room_detection', 28)}</div>
                <div style="color: #fff; font-weight: 600; margin-bottom: 4px;">Room Detection</div>
                <div style="color: #71717a; font-size: 12px;">AI identifies each room type</div>
            </div>
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('folder', 28)}</div>
                <div style="color: #fff; font-weight: 600; margin-bottom: 4px;">Auto Order</div>
                <div style="color: #71717a; font-size: 12px;">Standard delivery sequence</div>
            </div>
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('file', 28)}</div>
                <div style="color: #fff; font-weight: 600; margin-bottom: 4px;">Rename</div>
                <div style="color: #71717a; font-size: 12px;">Photo-1, Photo-2, Drone-1...</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Photo delivery order reference
        st.markdown(f"""
        <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                    border-radius: 12px; padding: 16px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                {icon('info', 16)}
                <span style="color: #fff; font-weight: 600; font-size: 14px;">Standard Photo Delivery Order</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 12px; color: #a1a1aa;">
                <div>1. Front of home / front patio</div>
                <div>6. Kitchen</div>
                <div>2. Entryway</div>
                <div>7. Master bedroom & bath</div>
                <div>3. Living room</div>
                <div>8. Secondary bedrooms</div>
                <div>4. Dining room</div>
                <div>9. Laundry area</div>
                <div>5. Family room</div>
                <div>10. Back of home / backyard / ADU</div>
            </div>
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(123, 140, 222, 0.2); color: #71717a; font-size: 11px;">
                Naming: Photo-1, Photo-2... | Drone-1, Drone-2... | Twilight-1, Twilight-2...
            </div>
        </div>
        """, unsafe_allow_html=True)

        # How it Works section for Photos
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                {icon('info', 18)}
                <span style="color: #fff; font-weight: 600; font-size: 15px;">How it Works</span>
            </div>
            <ol style="color: #a1a1aa; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.8;">
                <li>Paste your Dropbox folder link containing photos</li>
                <li>AI analyzes each image to detect the room type</li>
                <li>Photos are automatically sorted into standard delivery order</li>
                <li>Files are renamed (Photo-1, Photo-2, Drone-1, etc.) and packaged for download</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # Intelligent Room Detection breakdown
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
                {icon('room_detection', 18)}
                <span style="color: #fff; font-weight: 600; font-size: 15px;">Intelligent Room Detection</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                <div style="background: rgba(123, 140, 222, 0.1); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="color: #7B8CDE; font-size: 11px; font-weight: 600;">EXTERIOR</div>
                    <div style="color: #a1a1aa; font-size: 10px; margin-top: 4px;">Front · Rear</div>
                </div>
                <div style="background: rgba(123, 140, 222, 0.1); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="color: #7B8CDE; font-size: 11px; font-weight: 600;">LIVING SPACES</div>
                    <div style="color: #a1a1aa; font-size: 10px; margin-top: 4px;">Living · Dining · Family</div>
                </div>
                <div style="background: rgba(123, 140, 222, 0.1); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="color: #7B8CDE; font-size: 11px; font-weight: 600;">KITCHEN</div>
                    <div style="color: #a1a1aa; font-size: 10px; margin-top: 4px;">Kitchen · Pantry</div>
                </div>
                <div style="background: rgba(123, 140, 222, 0.1); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="color: #7B8CDE; font-size: 11px; font-weight: 600;">BEDROOMS</div>
                    <div style="color: #a1a1aa; font-size: 10px; margin-top: 4px;">Primary · Secondary</div>
                </div>
                <div style="background: rgba(123, 140, 222, 0.1); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="color: #7B8CDE; font-size: 11px; font-weight: 600;">BATHROOMS</div>
                    <div style="color: #a1a1aa; font-size: 10px; margin-top: 4px;">Full · Half · Primary</div>
                </div>
                <div style="background: rgba(123, 140, 222, 0.1); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="color: #7B8CDE; font-size: 11px; font-weight: 600;">UTILITY</div>
                    <div style="color: #a1a1aa; font-size: 10px; margin-top: 4px;">Laundry · Garage · Office</div>
                </div>
                <div style="background: rgba(123, 140, 222, 0.1); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="color: #7B8CDE; font-size: 11px; font-weight: 600;">OUTDOOR</div>
                    <div style="color: #a1a1aa; font-size: 10px; margin-top: 4px;">Yard · Pool · Patio</div>
                </div>
                <div style="background: rgba(123, 140, 222, 0.1); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="color: #7B8CDE; font-size: 11px; font-weight: 600;">SPECIAL</div>
                    <div style="color: #a1a1aa; font-size: 10px; margin-top: 4px;">ADU · Drone · Detail</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dropbox input for photos
        photo_sort_link = st.text_input(
            "Dropbox Folder Link",
            placeholder="Paste Dropbox folder link with photos to sort...",
            key="photo_sort_dropbox_autosort",
            label_visibility="collapsed"
        )

        # Analyze button - always visible
        analyze_clicked = st.button("🔍 Analyze & Preview Sort Order", key="btn_photo_sort_autosort", use_container_width=True, type="primary")

        if analyze_clicked and not photo_sort_link:
            st.warning("Please paste a Dropbox folder link first")

        if analyze_clicked and photo_sort_link:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.markdown("""
            <div style="color: #7B8CDE; font-size: 13px;">Downloading folder from Dropbox...</div>
            """, unsafe_allow_html=True)
            progress_bar.progress(0.1)

            tmp_path, filename, error = download_from_dropbox(photo_sort_link)

            if error:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error: {error}")
            elif tmp_path:
                is_zip = tmp_path.lower().endswith('.zip') or filename.lower().endswith('.zip')

                if not is_zip:
                    progress_bar.empty()
                    status_text.empty()
                    st.warning("Please provide a folder link (not a single file)")
                else:
                    status_text.markdown("""
                    <div style="color: #7B8CDE; font-size: 13px;">Extracting photos...</div>
                    """, unsafe_allow_html=True)
                    progress_bar.progress(0.2)

                    photo_paths = extract_zip_photos(tmp_path)

                    if not photo_paths:
                        progress_bar.empty()
                        status_text.empty()
                        st.warning("No photos found in folder")
                    else:
                        status_text.markdown(f"""
                        <div style="color: #7B8CDE; font-size: 13px;">Analyzing {len(photo_paths)} photos...</div>
                        """, unsafe_allow_html=True)

                        # Analyze each photo to detect room type
                        photos_data = []
                        for i, path in enumerate(photo_paths):
                            fname = os.path.basename(path)
                            progress = 0.2 + (0.6 * (i + 1) / len(photo_paths))
                            progress_bar.progress(min(progress, 0.8))

                            # Detect room type
                            room_type, confidence = classify_photo_room(path, fname)
                            photo_type = detect_photo_type(fname)

                            photos_data.append({
                                'filename': fname,
                                'path': path,
                                'room_type': room_type,
                                'original_room_type': room_type,  # Store original for learning
                                'room_confidence': confidence,
                                'photo_type': photo_type
                            })

                        # Sort photos
                        status_text.markdown("""
                        <div style="color: #7B8CDE; font-size: 13px;">Sorting photos...</div>
                        """, unsafe_allow_html=True)
                        progress_bar.progress(0.9)

                        sorted_photos = sort_photos_for_delivery(photos_data)

                        progress_bar.progress(1.0)
                        progress_bar.empty()
                        status_text.empty()

                        # Store in session state
                        st.session_state['photo_sort_data'] = sorted_photos
                        st.session_state['photo_sort_folder'] = filename

                        # Display preview
                        st.markdown(f"""
                        <div style="background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.3);
                                    border-radius: 10px; padding: 16px; margin-bottom: 20px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="color: #4ade80; font-size: 18px;">✓</span>
                                <span style="color: #4ade80; font-weight: 600;">Found {len(sorted_photos)} photos ready to sort</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Group by photo type
                        standard = [p for p in sorted_photos if p['photo_type'] == 'photo']
                        drones = [p for p in sorted_photos if p['photo_type'] == 'drone']
                        twilights = [p for p in sorted_photos if p['photo_type'] == 'twilight']

                        # Store photos in session state for persistence (with image data)
                        st.session_state['photo_sort_results'] = []
                        st.session_state['photo_sort_link'] = photo_sort_link

                        for p in sorted_photos:
                            photo_data = p.copy()
                            # Read and cache the image bytes for persistence
                            if os.path.exists(p['path']):
                                try:
                                    with open(p['path'], 'rb') as f:
                                        photo_data['image_bytes'] = f.read()
                                except:
                                    photo_data['image_bytes'] = None
                            else:
                                photo_data['image_bytes'] = None
                            st.session_state['photo_sort_results'].append(photo_data)

        # Display results if we have them in session state (OUTSIDE button block so it persists)
        if 'photo_sort_results' in st.session_state and st.session_state['photo_sort_results']:
            sorted_photos = st.session_state['photo_sort_results']
            photo_sort_link = st.session_state.get('photo_sort_link', '')

            st.markdown(f"""
            <div style="background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.3);
                        border-radius: 10px; padding: 16px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #4ade80; font-size: 18px;">✓</span>
                    <span style="color: #4ade80; font-weight: 600;">{len(sorted_photos)} photos ready to sort</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Display sorted preview with editable room types - SINGLE COLUMN
            st.markdown("### Preview & Edit Room Types")
            st.caption("Change room types below, then click 'Apply Changes & Re-sort' to update the order.")

            room_options = list(ROOM_TYPES.keys())
            room_labels = {k: ROOM_TYPES[k].get('name', k) for k in room_options}

            # Use a form to prevent reruns on every dropdown change
            with st.form(key="photo_edit_form"):
                # Single column layout - one photo per row
                for idx, p in enumerate(sorted_photos):
                    with st.container():
                        # Large thumbnail
                        if p.get('image_bytes'):
                            st.image(p['image_bytes'], width=350)
                        else:
                            st.markdown("📷 Image preview unavailable")

                        # Info row
                        col1, col2, col3 = st.columns([2, 2, 1])

                        with col1:
                            st.markdown(f"**Original:** {p['filename'][:30]}{'...' if len(p['filename']) > 30 else ''}")

                        with col2:
                            # Editable room type dropdown
                            current_room = p.get('room_type', 'living_room')
                            current_idx_room = room_options.index(current_room) if current_room in room_options else 0

                            st.selectbox(
                                "Room Type",
                                options=room_options,
                                index=current_idx_room,
                                format_func=lambda x: room_labels.get(x, x),
                                key=f"room_edit_{idx}",
                                label_visibility="collapsed"
                            )

                        with col3:
                            new_name = p.get('new_filename', f"Photo-{idx+1}")
                            st.markdown(f"**→ {new_name}**")

                        st.markdown("---")

                # Submit button inside the form
                submitted = st.form_submit_button("🔄 Apply Changes & Re-sort", use_container_width=True, type="primary")

            # Process form submission
            if submitted:
                num_photos = len(st.session_state['photo_sort_results'])
                progress_bar = st.progress(0)
                status = st.empty()
                corrections_saved = 0

                status.markdown(f"**Updating room types for {num_photos} photos...**")

                # First, update room types from form values and save corrections for learning
                for idx in range(num_photos):
                    form_key = f"room_edit_{idx}"
                    if form_key in st.session_state:
                        photo = st.session_state['photo_sort_results'][idx]
                        original_room = photo.get('original_room_type', photo.get('room_type'))
                        new_room = st.session_state[form_key]

                        # Save correction for learning if changed
                        if new_room != original_room and photo.get('image_bytes'):
                            try:
                                learning_db.save_correction(
                                    image_bytes=photo['image_bytes'],
                                    original_filename=photo.get('filename', ''),
                                    predicted_room=original_room,
                                    corrected_room=new_room
                                )
                                corrections_saved += 1
                            except Exception:
                                pass  # Don't fail if learning db has issues

                        st.session_state['photo_sort_results'][idx]['room_type'] = new_room
                    progress_bar.progress(0.1 + (0.15 * (idx + 1) / num_photos))

                if corrections_saved > 0:
                    status.markdown(f"**Learning from {corrections_saved} corrections...** 🧠")
                    progress_bar.progress(0.3)

                status.markdown(f"**Sorting {num_photos} photos...** (applying room flow order)")
                progress_bar.progress(0.4)

                # Re-sort based on updated room types
                resorted = sort_photos_for_delivery(st.session_state['photo_sort_results'])

                # Preserve image bytes and original predictions
                progress_bar.progress(0.6)
                status.markdown(f"**Preserving image data...** ({num_photos} photos)")
                for i, p in enumerate(resorted):
                    if i < len(st.session_state['photo_sort_results']):
                        # Find original by filename and copy image bytes
                        for orig in st.session_state['photo_sort_results']:
                            if orig['filename'] == p['filename']:
                                p['image_bytes'] = orig.get('image_bytes')
                                p['original_room_type'] = orig.get('original_room_type', orig.get('room_type'))
                                break
                    # Update progress
                    progress_bar.progress(0.6 + (0.35 * (i + 1) / num_photos))

                progress_bar.progress(1.0)
                if corrections_saved > 0:
                    status.markdown(f"**✅ Done!** Learned from {corrections_saved} corrections. Refreshing...")
                else:
                    status.markdown("**✅ Done!** Refreshing view...")
                st.session_state['photo_sort_results'] = resorted
                st.rerun()

            # Download/Save section (always visible when photos loaded)
            st.markdown("### Save Sorted Photos")

            save_col1, save_col2 = st.columns(2)

            with save_col1:
                # Save to Dropbox button with live progress
                if st.button("☁️ Save to Dropbox", key="btn_save_to_dropbox_photos", use_container_width=True, type="primary"):
                    import time as time_module

                    photos_to_save = st.session_state.get('photo_sort_results', sorted_photos)
                    num_photos = len(photos_to_save)

                    # Estimate time (~1-2 seconds per photo for Dropbox API)
                    est_seconds = max(5, int(num_photos * 1.5))

                    progress_bar = st.progress(0)
                    status = st.empty()
                    time_status = st.empty()

                    status.markdown(f"**☁️ Saving {num_photos} photos to Dropbox...**")
                    time_status.markdown(f"*Estimated time: ~{est_seconds} seconds*")

                    start_time = time_module.time()

                    # Progress callback for live updates
                    def update_progress(current, total, message):
                        if total > 0:
                            progress_bar.progress(current / total)
                        elapsed = time_module.time() - start_time
                        if current > 0:
                            remaining = (elapsed / current) * (total - current)
                            status.markdown(f"**☁️ {message}** ({current}/{total})")
                            time_status.markdown(f"*{int(remaining)}s remaining...*")
                        else:
                            status.markdown(f"**☁️ {message}**")

                    try:
                        # Get Dropbox client
                        dbx = dropbox.Dropbox(
                            oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
                            app_key=DROPBOX_APP_KEY,
                            app_secret=DROPBOX_APP_SECRET
                        )

                        # Save to Dropbox with progress callback
                        dest_folder, num_copied = organize_photos_in_dropbox(
                            dbx,
                            photo_sort_link,
                            photos_to_save,
                            progress_callback=update_progress
                        )

                        total_time = time_module.time() - start_time
                        progress_bar.progress(1.0)
                        status.empty()
                        time_status.empty()
                        st.success(f"✅ Saved {num_copied} photos to Dropbox in {total_time:.1f}s!")
                        st.info(f"📁 New folder: {dest_folder}")

                    except Exception as e:
                        progress_bar.empty()
                        status.empty()
                        time_status.empty()
                        st.error(f"Error saving to Dropbox: {str(e)}")

                st.caption("Creates a new '_Sorted' folder in Dropbox with renamed photos")

            with save_col2:
                # Download ZIP - one click prepare and download
                import zipfile
                import io
                import time

                # Check if ZIP is ready in session state
                if 'prepared_zip' not in st.session_state:
                    st.session_state['prepared_zip'] = None
                    st.session_state['zip_ready'] = False

                # Show download button if ZIP is already prepared
                if st.session_state.get('zip_ready') and st.session_state.get('prepared_zip'):
                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=st.session_state['prepared_zip'],
                        file_name="sorted_photos.zip",
                        mime="application/zip",
                        use_container_width=True,
                        type="primary"
                    )
                    st.caption("✅ ZIP ready! Click above to download")
                    # Button to prepare again if needed
                    if st.button("🔄 Re-prepare ZIP", key="btn_reprepare_zip", use_container_width=True):
                        st.session_state['zip_ready'] = False
                        st.rerun()
                else:
                    # One-click prepare and download button
                    if st.button("📦 Prepare & Download ZIP", key="btn_prepare_download_zip", use_container_width=True, type="primary"):
                        photos_to_zip = st.session_state.get('photo_sort_results', sorted_photos)
                        num_photos = len(photos_to_zip)

                        # Calculate estimated time (rough: ~0.1s per photo)
                        est_seconds = max(1, int(num_photos * 0.1))

                        progress_bar = st.progress(0)
                        status = st.empty()
                        time_status = st.empty()

                        status.markdown(f"**📦 Preparing {num_photos} photos for download...**")
                        time_status.markdown(f"*Estimated time: ~{est_seconds} seconds*")

                        start_time = time.time()
                        zip_buffer = io.BytesIO()

                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for i, p in enumerate(photos_to_zip):
                                # Use image_bytes if available (for cloud deployment)
                                if p.get('image_bytes'):
                                    _, ext = os.path.splitext(p['filename'])
                                    new_name = f"{p['new_filename']}{ext}"
                                    zf.writestr(new_name, p['image_bytes'])
                                elif os.path.exists(p.get('path', '')):
                                    _, ext = os.path.splitext(p['filename'])
                                    new_name = f"{p['new_filename']}{ext}"
                                    zf.write(p['path'], new_name)

                                # Update progress with live status
                                progress = (i + 1) / num_photos
                                progress_bar.progress(progress)
                                elapsed = time.time() - start_time
                                if i > 0:
                                    remaining = (elapsed / (i + 1)) * (num_photos - i - 1)
                                    status.markdown(f"**📦 Adding photo {i+1} of {num_photos}:** {p.get('new_filename', f'Photo-{i+1}')}")
                                    time_status.markdown(f"*{int(remaining)}s remaining...*")
                                else:
                                    status.markdown(f"**📦 Adding photo {i+1} of {num_photos}...**")

                        zip_buffer.seek(0)
                        st.session_state['prepared_zip'] = zip_buffer.getvalue()
                        st.session_state['zip_ready'] = True

                        total_time = time.time() - start_time
                        progress_bar.progress(1.0)
                        status.markdown(f"**✅ ZIP ready!** ({num_photos} photos in {total_time:.1f}s)")
                        time_status.empty()

                        # Rerun to show download button
                        st.rerun()

                    st.caption("Click to prepare and download your sorted photos")

        # Footer with stats
        render_footer()
        return  # End photo sorting mode here

    # =============================================
    # VIDEO SORTING MODE (Default)
    # =============================================
    # Feature overview cards for VIDEO
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 30px;">
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="margin-bottom: 8px;">{icon('room_detection', 28)}</div>
            <div style="color: #fff; font-weight: 600; margin-bottom: 4px;">Room Detection</div>
            <div style="color: #71717a; font-size: 12px;">Kitchen, bedroom, bathroom, living room, exterior, drone</div>
        </div>
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="margin-bottom: 8px;">{icon('best_moments', 28)}</div>
            <div style="color: #fff; font-weight: 600; margin-bottom: 4px;">Best Moments</div>
            <div style="color: #71717a; font-size: 12px;">Finds smoothest, best-exposed sections of each clip</div>
        </div>
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="margin-bottom: 8px;">{icon('xml_export', 28)}</div>
            <div style="color: #fff; font-weight: 600; margin-bottom: 4px;">XML Export</div>
            <div style="color: #71717a; font-size: 12px;">DaVinci Resolve, Premiere Pro, Final Cut Pro X</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for clips
    if 'auto_sort_clips' not in st.session_state:
        st.session_state.auto_sort_clips = []
    if 'auto_sort_analyzed' not in st.session_state:
        st.session_state.auto_sort_analyzed = False
    if 'dropbox_token' not in st.session_state:
        st.session_state.dropbox_token = os.environ.get('DROPBOX_ACCESS_TOKEN', '')
    if 'dropbox_files' not in st.session_state:
        st.session_state.dropbox_files = []
    if 'dropbox_shared_link' not in st.session_state:
        st.session_state.dropbox_shared_link = ''

    # Input method tabs
    st.markdown("### Select Footage Source")

    input_method = st.radio(
        "Input method",
        ["Dropbox Shared Link", "Upload Files", "Local Folder Path"],
        horizontal=True,
        label_visibility="collapsed"
    )

    clips_to_analyze = []
    original_filenames = {}  # Map temp paths to original Dropbox filenames

    if input_method == "Dropbox Shared Link":
        # =============================================
        # CHECK FOR REVIEW STATE FIRST - Show review UI if clips are ready
        # =============================================
        if st.session_state.get('dropbox_clips_for_review'):
            clips_for_review = st.session_state.dropbox_clips_for_review

            # Show any processing errors from the analysis phase
            if st.session_state.get('dropbox_processing_errors'):
                errors = st.session_state.dropbox_processing_errors
                with st.expander(f"{len(errors)} clip(s) could not be processed (click to see details)", expanded=False):
                    for err in errors[:20]:
                        st.text(err)
                    if len(errors) > 20:
                        st.info(f"... and {len(errors) - 20} more")

            st.markdown("""
            <div style="background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.3);
                        border-radius: 12px; padding: 20px; margin: 20px 0;">
                <div style="color: #4ade80; font-size: 18px; font-weight: 700; margin-bottom: 8px;">
                    Review & Edit Room Assignments
                </div>
                <div style="color: #a1a1aa; font-size: 13px;">
                    Analysis complete! Review the detected room types below. Fix any incorrect ones, then click "Create Organized Folder".
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Room type options for dropdown
            room_options = list(ROOM_TYPES.keys())

            st.markdown(f"**{len(clips_for_review)} clips analyzed**")

            # Create editable list
            updated_clips = []
            for idx, clip in enumerate(clips_for_review):
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.text(clip['filename'][:40] + ('...' if len(clip['filename']) > 40 else ''))
                with col2:
                    current_room = clip['room_type']
                    current_idx = room_options.index(current_room) if current_room in room_options else 0
                    new_room = st.selectbox(
                        f"Room {idx}",
                        room_options,
                        index=current_idx,
                        key=f"review_room_{idx}",
                        label_visibility="collapsed"
                    )
                    clip['room_type'] = new_room
                with col3:
                    conf = clip.get('room_confidence', 0)
                    color = "#4ade80" if conf > 0.7 else "#fbbf24" if conf > 0.4 else "#ef4444"
                    st.markdown(f"<span style='color: {color};'>{conf:.0%}</span>", unsafe_allow_html=True)
                updated_clips.append(clip)

            st.markdown("---")

            # Action buttons
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("Create Organized Folder in Dropbox", type="primary", use_container_width=True):
                    try:
                        dbx = dropbox.Dropbox(
                            oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
                            app_key=DROPBOX_APP_KEY,
                            app_secret=DROPBOX_APP_SECRET
                        )
                        with st.spinner("Creating organized folder..."):
                            source_link = st.session_state.get('dropbox_source_link', '')
                            project_name = st.session_state.get('dropbox_project_name', 'AC_Project')
                            export_fmt = st.session_state.get('dropbox_export_format', 'DaVinci Resolve')

                            link_meta = dbx.sharing_get_shared_link_metadata(source_link)
                            source_path = link_meta.path_lower if hasattr(link_meta, 'path_lower') else ""

                            dest_folder, num_clips, xml_filename = organize_in_dropbox(
                                dbx, source_path, project_name, updated_clips, export_fmt
                            )

                            st.session_state.dropbox_clips_for_review = None
                            st.success(f"""
                            **Done!** Organized folder created:

                            **{dest_folder}**

                            - {num_clips} clips organized
                            - {xml_filename} timeline ready

                            Check your Dropbox!
                            """)
                    except Exception as e:
                        st.error(f"Error: {e}")
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.dropbox_clips_for_review = None
                    st.rerun()

        else:
            # Normal input flow - show when NOT reviewing
            # Simple Dropbox integration - no authentication needed
            st.markdown("""
            <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                        border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                <div style="color: #7B8CDE; font-weight: 600; margin-bottom: 8px;">One-Click Workflow</div>
                <ol style="color: #a1a1aa; font-size: 13px; margin: 0; padding-left: 20px;">
                    <li>Paste your Dropbox folder link</li>
                    <li>Choose where to save (Desktop, etc.)</li>
                    <li>Click "Go" - we handle the rest</li>
                    <li>Open your folder with organized clips + XML</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

            # Shared link input
            dropbox_link = st.text_input(
            "Dropbox Shared Folder Link",
            value=st.session_state.get('dropbox_shared_link', ''),
            placeholder="https://www.dropbox.com/scl/fo/...",
            help="Paste the shared link to your raw footage folder"
        )

        if dropbox_link:
            st.session_state.dropbox_shared_link = dropbox_link
            link_info = parse_dropbox_shared_link(dropbox_link)

            if not link_info["is_valid"]:
                st.error("Invalid Dropbox link. Please paste a valid Dropbox shared link.")
            elif not link_info["is_folder"]:
                st.warning("This looks like a file link. Please share the **folder** containing your clips.")
            else:
                st.success("Valid Dropbox folder link")

                # =============================================
                # PROJECT SETTINGS (Simplified)
                # =============================================
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                # Project name input
                project_name_input = st.text_input(
                    "Project folder name",
                    value="AC_Project",
                    help="Name for the downloaded project folder",
                    placeholder="Enter project name..."
                )

                # Export format
                export_format = st.selectbox(
                    "Export format",
                    ["DaVinci Resolve", "Adobe Premiere Pro", "Final Cut Pro X"],
                    help="Choose your editing software"
                )

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                # Dropbox API mode toggle
                organize_in_dropbox = st.checkbox(
                    "Organize directly in Dropbox (recommended for large projects)",
                    value=st.session_state.get('organize_in_dropbox', False),
                    help="Instead of downloading files, creates an organized folder directly in your Dropbox. Best for projects over 5GB."
                )
                st.session_state.organize_in_dropbox = organize_in_dropbox

                if organize_in_dropbox:
                    # Show Dropbox connection status (pre-authorized for Aerial Canvas team)
                    try:
                        dbx = dropbox.Dropbox(
                            oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
                            app_key=DROPBOX_APP_KEY,
                            app_secret=DROPBOX_APP_SECRET
                        )
                        account = dbx.users_get_current_account()
                        st.success(f"Connected to **{account.name.display_name}** ({account.email})")
                        # Store connection status in session state
                        st.session_state.dropbox_connected = True
                    except Exception as e:
                        dbx = None
                        st.session_state.dropbox_connected = False
                        st.error(f"Could not connect to Dropbox: {e}")
                else:
                    st.session_state.dropbox_connected = False

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                # Analysis mode selection - clickable cards
                st.markdown("<div style='color: #a1a1aa; font-size: 13px; margin: 8px 0 12px 0;'>Analysis mode:</div>", unsafe_allow_html=True)

                # Initialize mode in session state
                if 'analysis_mode_dropbox' not in st.session_state:
                    st.session_state.analysis_mode_dropbox = 'timeline_only'

                mode_col1, mode_col2, mode_col3 = st.columns(3)

                with mode_col1:
                    is_selected = st.session_state.analysis_mode_dropbox == 'timeline_only'
                    border_color = "#4ade80" if is_selected else "#1d1d1f"
                    bg_color = "rgba(74, 222, 128, 0.08)" if is_selected else "#111"
                    st.markdown(f"""
                    <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; padding: 20px 16px 24px 16px; text-align: center;">
                        <div style="margin-bottom: 10px;">{icon('mode_timeline', 28)}</div>
                        <div style="color: #fff; font-weight: 600; font-size: 14px; margin-bottom: 8px;">Timeline Only</div>
                        <div style="color: #71717a; font-size: 11px; line-height: 1.5;">Fastest (~1s/clip)<br>Room sorting + auto-structure<br>Uses full clips</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Select", key="mode_timeline", use_container_width=True, type="primary" if is_selected else "secondary"):
                        st.session_state.analysis_mode_dropbox = 'timeline_only'
                        st.rerun()

                with mode_col2:
                    is_selected = st.session_state.analysis_mode_dropbox == 'quick'
                    border_color = "#4ade80" if is_selected else "#1d1d1f"
                    bg_color = "rgba(74, 222, 128, 0.08)" if is_selected else "#111"
                    st.markdown(f"""
                    <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; padding: 20px 16px 24px 16px; text-align: center;">
                        <div style="margin-bottom: 10px;">{icon('mode_quick', 28)}</div>
                        <div style="color: #fff; font-weight: 600; font-size: 14px; margin-bottom: 8px;">Quick Mode</div>
                        <div style="color: #71717a; font-size: 11px; line-height: 1.5;">Fast (~2s/clip)<br>Adds trim detection<br>15% in/out heuristic</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Select", key="mode_quick", use_container_width=True, type="primary" if is_selected else "secondary"):
                        st.session_state.analysis_mode_dropbox = 'quick'
                        st.rerun()

                with mode_col3:
                    is_selected = st.session_state.analysis_mode_dropbox == 'full'
                    border_color = "#4ade80" if is_selected else "#1d1d1f"
                    bg_color = "rgba(74, 222, 128, 0.08)" if is_selected else "#111"
                    st.markdown(f"""
                    <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; padding: 20px 16px 24px 16px; text-align: center;">
                        <div style="margin-bottom: 10px;">{icon('mode_full', 28)}</div>
                        <div style="color: #fff; font-weight: 600; font-size: 14px; margin-bottom: 8px;">Full Analysis</div>
                        <div style="color: #71717a; font-size: 11px; line-height: 1.5;">Detailed (~15s/clip)<br>Precise trim points<br>Optical flow analysis</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Select", key="mode_full", use_container_width=True, type="primary" if is_selected else "secondary"):
                        st.session_state.analysis_mode_dropbox = 'full'
                        st.rerun()

                analysis_mode = st.session_state.analysis_mode_dropbox
                quick_mode = analysis_mode in ["quick", "timeline_only"]
                timeline_only = analysis_mode == "timeline_only"

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # Check if we should use Dropbox API mode
                    use_dropbox_api = st.session_state.get('organize_in_dropbox', False) and st.session_state.get('dropbox_connected', False)

                    # Disable Go button if Dropbox API mode is selected but not connected
                    go_disabled = st.session_state.get('organize_in_dropbox', False) and not st.session_state.get('dropbox_connected', False)

                    if go_disabled:
                        st.warning("Please connect to Dropbox first to use this mode.")

                    # Single "Go" button that does everything
                    if st.button("Go", type="primary", use_container_width=True, disabled=go_disabled):
                        st.session_state.auto_sort_clips = []
                        st.session_state.quick_mode = quick_mode
                        st.session_state.timeline_only = timeline_only
                        st.session_state.project_name_for_download = project_name_input
                        st.session_state.export_format_for_download = export_format

                        # =============================================
                        # DROPBOX API MODE - Organize directly in Dropbox
                        # =============================================
                        if use_dropbox_api:
                            # Create Dropbox client directly (don't use get_dropbox_client which has issues)
                            dbx = dropbox.Dropbox(
                                oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
                                app_key=DROPBOX_APP_KEY,
                                app_secret=DROPBOX_APP_SECRET
                            )
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            try:
                                status_text.markdown("**Step 1/4: Getting folder contents from Dropbox...**")
                                progress_bar.progress(0.1)

                                # Get shared link metadata to find the folder path
                                shared_link = dropbox.files.SharedLink(url=dropbox_link)

                                # List files in the shared folder
                                result = dbx.files_list_folder(path="", shared_link=shared_link)
                                all_files = result.entries

                                # Get more files if there are more
                                while result.has_more:
                                    result = dbx.files_list_folder_continue(result.cursor)
                                    all_files.extend(result.entries)

                                # Filter for video files
                                video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.m4v'}
                                video_files = [f for f in all_files
                                              if isinstance(f, FileMetadata)
                                              and os.path.splitext(f.name)[1].lower() in video_extensions]

                                if not video_files:
                                    st.error("No video files found in the Dropbox folder.")
                                else:
                                    # Debug: Show first file's info
                                    first_file = video_files[0]
                                    debug_info = f"First file: {first_file.name}"
                                    debug_info += f", id: {getattr(first_file, 'id', 'None')}"
                                    debug_info += f", path_lower: {getattr(first_file, 'path_lower', 'None')}"
                                    st.info(f"Debug: {debug_info}")

                                    status_text.markdown(f"**Step 2/4: Analyzing {len(video_files)} clips...**")
                                    progress_bar.progress(0.2)

                                    # Analyze each clip (download temporarily for analysis)
                                    clips_data = []
                                    processing_errors = []  # Store errors to show after processing
                                    temp_dir = tempfile.mkdtemp()

                                    for idx, file_meta in enumerate(video_files):
                                        progress_bar.progress(0.2 + (idx + 1) / len(video_files) * 0.5)
                                        status_text.markdown(f"**Analyzing clip {idx + 1} of {len(video_files)}:** {file_meta.name[:40]}...")

                                        # Download file temporarily for analysis
                                        temp_path = os.path.join(temp_dir, file_meta.name)
                                        try:
                                            # Download using shared link
                                            file_downloaded = False
                                            download_errors = []

                                            # Get file ID if available (most reliable for newer Dropbox links)
                                            file_id = getattr(file_meta, 'id', None)

                                            # Method 1: Try using file ID with files_download
                                            if file_id and not file_downloaded:
                                                try:
                                                    metadata, response = dbx.files_download(file_id)
                                                    with open(temp_path, 'wb') as f:
                                                        f.write(response.content)
                                                    file_downloaded = True
                                                except Exception as e:
                                                    download_errors.append(f"file_id method: {str(e)[:80]}")

                                            # Method 2: Try shared link with various path formats
                                            if not file_downloaded:
                                                paths_to_try = [
                                                    f"/{file_meta.name}",
                                                    file_meta.name,
                                                    None,  # No path
                                                ]
                                                for try_path in paths_to_try:
                                                    if file_downloaded:
                                                        break
                                                    try:
                                                        if try_path is None:
                                                            metadata, response = dbx.sharing_get_shared_link_file(url=dropbox_link)
                                                        else:
                                                            metadata, response = dbx.sharing_get_shared_link_file(url=dropbox_link, path=try_path)
                                                        with open(temp_path, 'wb') as f:
                                                            f.write(response.content)
                                                        file_downloaded = True
                                                    except Exception as e:
                                                        download_errors.append(f"path '{try_path}': {str(e)[:60]}")

                                            # Method 3: Try direct download URL construction
                                            if not file_downloaded:
                                                try:
                                                    # Convert shared link to direct download link
                                                    direct_url = dropbox_link.replace("www.dropbox.com", "dl.dropboxusercontent.com")
                                                    if "?" in direct_url:
                                                        direct_url = direct_url.split("?")[0]
                                                    direct_url = f"{direct_url}/{file_meta.name}?dl=1"

                                                    response = requests.get(direct_url, timeout=120)
                                                    if response.status_code == 200:
                                                        with open(temp_path, 'wb') as f:
                                                            f.write(response.content)
                                                        file_downloaded = True
                                                    else:
                                                        download_errors.append(f"direct URL: HTTP {response.status_code}")
                                                except Exception as e:
                                                    download_errors.append(f"direct URL: {str(e)[:60]}")

                                            if not file_downloaded:
                                                raise Exception(f"All methods failed: {'; '.join(download_errors[:3])}")

                                            # Classify room type
                                            room_type, confidence = classify_clip_room(temp_path, file_meta.name)

                                            # Get duration using ffprobe (faster than OpenCV for this)
                                            duration = 10  # Default
                                            fps = 30  # Default
                                            try:
                                                # Use ffprobe to get duration from header (very fast)
                                                probe_result = subprocess.run(
                                                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                                     '-show_entries', 'stream=duration,r_frame_rate',
                                                     '-of', 'json', temp_path],
                                                    capture_output=True, text=True, timeout=10
                                                )
                                                if probe_result.returncode == 0:
                                                    probe_data = json.loads(probe_result.stdout)
                                                    if probe_data.get('streams'):
                                                        stream = probe_data['streams'][0]
                                                        if 'duration' in stream:
                                                            duration = float(stream['duration'])
                                                        if 'r_frame_rate' in stream:
                                                            fps_parts = stream['r_frame_rate'].split('/')
                                                            if len(fps_parts) == 2 and int(fps_parts[1]) > 0:
                                                                fps = int(fps_parts[0]) / int(fps_parts[1])
                                            except:
                                                # Fallback to OpenCV
                                                try:
                                                    import cv2
                                                    cap = cv2.VideoCapture(temp_path)
                                                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                                                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                                    duration = frame_count / fps if fps > 0 else 10
                                                    cap.release()
                                                except:
                                                    pass

                                            quality = {"duration": duration, "fps": fps}
                                            if not timeline_only:
                                                quality = analyze_clip_quality(temp_path, quick_mode=quick_mode)

                                            clips_data.append({
                                                "filename": file_meta.name,
                                                "dropbox_path": f"/{file_meta.name}",
                                                "room_type": room_type,
                                                "room_confidence": confidence,
                                                "duration": quality.get("duration", duration),
                                                "fps": quality.get("fps", fps),
                                                "in_point": quality.get("best_in_point", 0),
                                                "out_point": quality.get("best_out_point", quality.get("duration", 10)),
                                            })

                                            # Delete temp file immediately
                                            os.remove(temp_path)

                                        except Exception as e:
                                            # Store error for display after processing completes (full error for debugging)
                                            processing_errors.append(f"{file_meta.name}: {str(e)}")
                                            if os.path.exists(temp_path):
                                                os.remove(temp_path)

                                    # Clean up temp directory
                                    try:
                                        os.rmdir(temp_dir)
                                    except:
                                        pass

                                    # Store errors in session state so they persist after rerun
                                    st.session_state.dropbox_processing_errors = processing_errors

                                    progress_bar.progress(1.0)
                                    status_text.empty()
                                    progress_bar.empty()

                                    # Check if we got any clips
                                    if clips_data:
                                        # Store data for review step
                                        st.session_state.dropbox_clips_for_review = clips_data
                                        st.session_state.dropbox_source_link = dropbox_link
                                        st.session_state.dropbox_project_name = project_name_input
                                        st.session_state.dropbox_export_format = export_format
                                        st.rerun()
                                    else:
                                        # Show detailed error info
                                        st.error(f"No clips could be analyzed. Found {len(video_files)} video files but none could be processed.")
                                        if processing_errors:
                                            st.markdown("**Errors encountered:**")
                                            for err in processing_errors[:10]:  # Show first 10 errors
                                                st.code(err)
                                            if len(processing_errors) > 10:
                                                st.info(f"... and {len(processing_errors) - 10} more errors")
                                        st.info("This is usually caused by the shared link not having the right permissions. Make sure the link allows 'Anyone with the link' to view/edit.")

                            except AuthError:
                                st.error("Dropbox authentication failed. Please reconnect.")
                                st.session_state.dropbox_access_token = None
                            except ApiError as e:
                                st.error(f"Dropbox API error: {e}")
                            except Exception as e:
                                st.error(f"Error: {e}")

                # =============================================
                # REVIEW STEP - Show analyzed clips for review/editing
                # =============================================
                if st.session_state.get('dropbox_clips_for_review'):
                    clips_for_review = st.session_state.dropbox_clips_for_review

                    # Show any processing errors from the analysis phase
                    if st.session_state.get('dropbox_processing_errors'):
                        errors = st.session_state.dropbox_processing_errors
                        with st.expander(f"⚠️ {len(errors)} clip(s) could not be processed", expanded=False):
                            for err in errors[:20]:
                                st.text(err)
                            if len(errors) > 20:
                                st.info(f"... and {len(errors) - 20} more")

                    st.markdown("""
                    <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                                border-radius: 12px; padding: 20px; margin: 20px 0;">
                        <div style="color: #7B8CDE; font-size: 18px; font-weight: 700; margin-bottom: 8px;">
                            Review & Edit Room Assignments
                        </div>
                        <div style="color: #a1a1aa; font-size: 13px;">
                            Check the detected room types below. Fix any incorrect ones before creating the organized folder.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Room type options for dropdown
                    room_options = list(ROOM_TYPES.keys())

                    # Create editable table
                    st.markdown("### Clips to Organize")

                    # Header
                    header_cols = st.columns([3, 2, 1])
                    with header_cols[0]:
                        st.markdown("**Filename**")
                    with header_cols[1]:
                        st.markdown("**Room Type**")
                    with header_cols[2]:
                        st.markdown("**Confidence**")

                    st.markdown("---")

                    # Editable rows
                    updated_clips = []
                    for idx, clip in enumerate(clips_for_review):
                        row_cols = st.columns([3, 2, 1])
                        with row_cols[0]:
                            st.markdown(f"`{clip['filename'][:35]}{'...' if len(clip['filename']) > 35 else ''}`")
                        with row_cols[1]:
                            # Dropdown to change room type
                            current_room = clip['room_type']
                            current_idx = room_options.index(current_room) if current_room in room_options else 0
                            new_room = st.selectbox(
                                f"Room for clip {idx}",
                                room_options,
                                index=current_idx,
                                key=f"room_select_{idx}",
                                label_visibility="collapsed"
                            )
                            clip['room_type'] = new_room
                        with row_cols[2]:
                            conf = clip.get('room_confidence', 0)
                            color = "#4ade80" if conf > 0.7 else "#fbbf24" if conf > 0.4 else "#ef4444"
                            st.markdown(f"<span style='color: {color};'>{conf:.0%}</span>", unsafe_allow_html=True)
                        updated_clips.append(clip)

                    st.markdown("---")

                    # Summary
                    room_counts = {}
                    for clip in updated_clips:
                        room = clip['room_type']
                        room_counts[room] = room_counts.get(room, 0) + 1

                    st.markdown("**Summary:** " + ", ".join([f"{ROOM_TYPES.get(r, {}).get('name', r)}: {c}" for r, c in sorted(room_counts.items())]))

                    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

                    # Action buttons
                    btn_col1, btn_col2 = st.columns([3, 1])
                    with btn_col1:
                        if st.button("Create Organized Folder in Dropbox", type="primary", use_container_width=True):
                            dbx = get_dropbox_client()
                            if dbx:
                                with st.spinner("Creating organized folder in Dropbox..."):
                                    try:
                                        # Get source path
                                        source_link = st.session_state.dropbox_source_link
                                        project_name = st.session_state.dropbox_project_name
                                        export_fmt = st.session_state.dropbox_export_format

                                        link_meta = dbx.sharing_get_shared_link_metadata(source_link)
                                        source_path = link_meta.path_lower if hasattr(link_meta, 'path_lower') else ""

                                        # Create organized folder with user-edited room assignments
                                        dest_folder, num_clips, xml_filename = organize_in_dropbox(
                                            dbx, source_path, project_name, updated_clips, export_fmt
                                        )

                                        # Clear review state
                                        st.session_state.dropbox_clips_for_review = None

                                        st.success(f"""
                                        **Done!** Organized folder created in your Dropbox:

                                        **{dest_folder}**

                                        - {num_clips} clips renamed and organized by room type
                                        - {xml_filename} timeline ready to import

                                        Open your Dropbox app or dropbox.com to sync the folder.
                                        """)
                                    except Exception as e:
                                        st.error(f"Error creating folder: {e}")
                    with btn_col2:
                        if st.button("Cancel", use_container_width=True):
                            st.session_state.dropbox_clips_for_review = None
                            st.rerun()

                # =============================================
                # DOWNLOAD MODE - Only if not using Dropbox API and not reviewing
                # =============================================
                if not st.session_state.get('organize_in_dropbox', False) and not st.session_state.get('dropbox_clips_for_review'):
                    if st.button("Go", type="primary", use_container_width=True, key="go_download_mode"):
                        # Create download URL (change dl=0 to dl=1)
                            download_url = dropbox_link.replace("dl=0", "dl=1")
                            if "dl=1" not in download_url:
                                if "?" in download_url:
                                    download_url += "&dl=1"
                                else:
                                    download_url += "?dl=1"

                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            time_text = st.empty()
                            cancel_placeholder = st.empty()

                            # Initialize cancel state
                            st.session_state.analysis_cancelled = False

                            # Show cancel button
                            def show_cancel_button():
                                if cancel_placeholder.button("Cancel", key="cancel_analysis", type="secondary"):
                                    st.session_state.analysis_cancelled = True
                                    return True
                                return False

                            show_cancel_button()

                            try:
                                # Download ZIP with time tracking
                                download_start = time.time()

                                status_text.markdown("""
                                <div style="margin-bottom: 8px;">
                                    <strong>Step 1/3: Downloading from Dropbox...</strong>
                                </div>
                                """, unsafe_allow_html=True)
                                progress_bar.progress(0.05)

                                temp_dir = tempfile.mkdtemp()
                                zip_path = os.path.join(temp_dir, "dropbox_folder.zip")

                                # Stream download with progress and time estimation
                                response = requests.get(download_url, stream=True, timeout=600)
                                response.raise_for_status()

                                total_size = int(response.headers.get('content-length', 0))
                                downloaded = 0
                                last_update = time.time()

                                # Format file size for display
                                def format_size(bytes):
                                    if bytes >= 1024 * 1024 * 1024:
                                        return f"{bytes / (1024**3):.1f} GB"
                                    elif bytes >= 1024 * 1024:
                                        return f"{bytes / (1024**2):.1f} MB"
                                    elif bytes >= 1024:
                                        return f"{bytes / 1024:.1f} KB"
                                    return f"{bytes} B"

                                with open(zip_path, 'wb') as f:
                                    for chunk in response.iter_content(chunk_size=32768):
                                        # Check for cancellation
                                        if st.session_state.get('analysis_cancelled', False):
                                            f.close()
                                            raise Exception("Cancelled by user")

                                        f.write(chunk)
                                        downloaded += len(chunk)

                                        # Update progress every 0.3 seconds to avoid UI lag
                                        current_time = time.time()
                                        if current_time - last_update > 0.3 or downloaded == total_size:
                                            last_update = current_time

                                            if total_size > 0:
                                                pct = min(0.4, 0.05 + (downloaded / total_size) * 0.35)
                                                progress_bar.progress(pct)

                                                # Calculate time estimates
                                                elapsed = current_time - download_start
                                                if downloaded > 0 and elapsed > 0:
                                                    speed = downloaded / elapsed
                                                    remaining_bytes = total_size - downloaded
                                                    eta_seconds = remaining_bytes / speed if speed > 0 else 0

                                                    # Format times
                                                    elapsed_fmt = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
                                                    eta_fmt = f"{int(eta_seconds // 60)}:{int(eta_seconds % 60):02d}"
                                                    speed_fmt = format_size(int(speed)) + "/s"

                                                    time_text.markdown(f"""
                                                    <div style="text-align: center; color: #a1a1aa; font-size: 12px; margin-top: 8px;">
                                                        {format_size(downloaded)} / {format_size(total_size)} &nbsp;|&nbsp;
                                                        {speed_fmt} &nbsp;|&nbsp;
                                                        Elapsed: {elapsed_fmt} &nbsp;|&nbsp;
                                                        ETA: {eta_fmt}
                                                    </div>
                                                    """, unsafe_allow_html=True)

                                download_time = time.time() - download_start
                                download_fmt = f"{int(download_time // 60)}:{int(download_time % 60):02d}"

                                progress_bar.progress(0.4)
                                time_text.markdown(f"""
                                <div style="text-align: center; color: #4ade80; font-size: 12px; margin-top: 8px;">
                                    Download complete ({format_size(total_size)}) in {download_fmt}
                                </div>
                                """, unsafe_allow_html=True)
                                status_text.markdown("**Step 2/3: Extracting files...**")

                                # Extract ZIP
                                extract_dir = os.path.join(temp_dir, "extracted")
                                os.makedirs(extract_dir, exist_ok=True)

                                with zipfile.ZipFile(zip_path, 'r') as zf:
                                    zf.extractall(extract_dir)

                                progress_bar.progress(0.5)

                                # Find all video files
                                video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.m4v'}
                                video_files = []
                                for root, dirs, files in os.walk(extract_dir):
                                    for f in files:
                                        ext = os.path.splitext(f)[1].lower()
                                        if ext in video_extensions:
                                            video_files.append(os.path.join(root, f))

                                if not video_files:
                                    st.error("No video files found in the Dropbox folder.")
                                else:
                                    status_text.markdown(f"**Step 2/3: Analyzing {len(video_files)} clips...**")

                                # Store temp_dir path for later export
                                st.session_state.auto_sort_temp_dir = extract_dir

                                # Time tracking for estimates
                                clip_times = []
                                analysis_start = time.time()

                                # Analyze each clip
                                for idx, clip_path in enumerate(video_files):
                                    # Check for cancellation
                                    if st.session_state.get('analysis_cancelled', False):
                                        raise Exception("Cancelled by user")

                                    clip_start = time.time()
                                    filename = os.path.basename(clip_path)

                                    # Calculate time estimates
                                    elapsed = time.time() - analysis_start
                                    elapsed_fmt = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"

                                    if clip_times:
                                        avg_time = sum(clip_times) / len(clip_times)
                                        remaining_clips = len(video_files) - idx
                                        eta_seconds = avg_time * remaining_clips
                                        eta_fmt = f"{int(eta_seconds // 60)}:{int(eta_seconds % 60):02d}"
                                    else:
                                        eta_fmt = "calculating..."

                                    status_text.markdown(f"**Analyzing clip {idx + 1} of {len(video_files)}:** {filename[:40]}{'...' if len(filename) > 40 else ''}")
                                    time_text.markdown(f"""
                                    <div style="text-align: center; color: #a1a1aa; font-size: 12px; margin-top: 8px;">
                                        Elapsed: {elapsed_fmt} &nbsp;|&nbsp; ETA: {eta_fmt} &nbsp;|&nbsp; Avg: {f'{sum(clip_times)/len(clip_times):.1f}s/clip' if clip_times else '...'}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    progress_bar.progress(0.5 + (idx + 1) / len(video_files) * 0.5)

                                    # Classify room type
                                    room_type, confidence = classify_clip_room(clip_path, filename)

                                    # Timeline Only mode: skip quality analysis, use full clip
                                    if timeline_only:
                                        # Just get basic duration from metadata
                                        try:
                                            import cv2
                                            cap = cv2.VideoCapture(clip_path)
                                            fps = cap.get(cv2.CAP_PROP_FPS) or 30
                                            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                            duration = frame_count / fps if fps > 0 else 10
                                            cap.release()
                                        except:
                                            duration = 10
                                            fps = 30

                                        quality = {
                                            "duration": duration,
                                            "fps": fps,
                                            "best_in_point": 0,
                                            "best_out_point": duration,
                                            "avg_stability": 0.8,
                                            "avg_exposure": 0.7,
                                            "best_segment_score": 0.7,
                                            "shake_start": 0,
                                            "shake_end": 0,
                                            "hero_duration": duration,
                                            "trim_suggestion": "Full clip (Timeline Only mode)"
                                        }
                                    else:
                                        # Analyze quality for best moments
                                        quality = analyze_clip_quality(clip_path, quick_mode=quick_mode)

                                    st.session_state.auto_sort_clips.append({
                                        "path": clip_path,
                                        "filename": filename,  # Original filename for XML
                                        "room_type": room_type,
                                        "room_confidence": confidence,
                                        "duration": quality.get("duration", 0),
                                        "fps": quality.get("fps", 30),
                                        "in_point": quality.get("best_in_point", 0),
                                        "out_point": quality.get("best_out_point", quality.get("duration", 10)),
                                        "stability_score": quality.get("avg_stability", 0),
                                        "exposure_score": quality.get("avg_exposure", 0),
                                        "segment_score": quality.get("best_segment_score", 0),
                                        "shake_start": quality.get("shake_start", 0),
                                        "shake_end": quality.get("shake_end", 0),
                                        "hero_duration": quality.get("hero_duration", quality.get("duration", 0)),
                                        "trim_suggestion": quality.get("trim_suggestion", "")
                                    })

                                    # Track time for this clip
                                    clip_times.append(time.time() - clip_start)

                                # Show completion stats
                                total_time = time.time() - analysis_start
                                total_fmt = f"{int(total_time // 60)}:{int(total_time % 60):02d}"
                                avg_per_clip = total_time / len(video_files) if video_files else 0

                                status_text.markdown(f"**Step 2/3: Analysis complete!** {len(video_files)} clips processed")
                                progress_bar.progress(0.7)

                                # =============================================
                                # CREATE DOWNLOADABLE ZIP
                                # =============================================
                                project_name = st.session_state.get('project_name_for_download', 'AC_Project')
                                export_fmt = st.session_state.get('export_format_for_download', 'DaVinci Resolve')

                                status_text.markdown(f"**Step 3/3: Creating {project_name}.zip...**")

                                import shutil
                                import io

                                # Create temp project folder
                                project_temp_dir = os.path.join(temp_dir, project_name)
                                os.makedirs(project_temp_dir, exist_ok=True)

                                # Group and sort clips by room type
                                clips_to_save = st.session_state.auto_sort_clips
                                room_order = list(ROOM_TYPES.keys())
                                sorted_clips = sorted(clips_to_save,
                                    key=lambda c: (room_order.index(c['room_type']) if c['room_type'] in room_order else 99, c['filename']))

                                # Copy and rename files
                                exported_clips = []
                                room_counters = {}

                                for idx, clip in enumerate(sorted_clips):
                                    progress_bar.progress(0.7 + (idx + 1) / len(sorted_clips) * 0.2)

                                    room = clip['room_type']
                                    room_name = ROOM_TYPES.get(room, {}).get('name', room).replace('/', '-').replace(' ', '')

                                    # Track counter per room
                                    if room not in room_counters:
                                        room_counters[room] = 1
                                    else:
                                        room_counters[room] += 1

                                    # Generate new filename: 01_Kitchen_01.mp4
                                    ext = os.path.splitext(clip['filename'])[1]
                                    new_filename = f"{idx+1:02d}_{room_name}_{room_counters[room]:02d}{ext}"
                                    dest_path = os.path.join(project_temp_dir, new_filename)

                                    # Copy file
                                    try:
                                        if os.path.exists(clip['path']):
                                            shutil.copy2(clip['path'], dest_path)
                                            exported_clip = clip.copy()
                                            exported_clip['path'] = new_filename  # Relative path for XML
                                            exported_clip['filename'] = new_filename
                                            exported_clips.append(exported_clip)
                                    except Exception as copy_err:
                                        pass  # Skip files that can't be copied

                                # Generate XML (outside the for loop)
                                if export_fmt == "Final Cut Pro X":
                                    xml_content = generate_fcpxml(exported_clips, project_name)
                                    xml_filename = f"{project_name}.fcpxml"
                                else:
                                    xml_content = generate_premiere_xml(exported_clips, project_name)
                                    xml_filename = f"{project_name}.xml"

                                xml_path = os.path.join(project_temp_dir, xml_filename)
                                with open(xml_path, 'w') as f:
                                    f.write(xml_content)

                                progress_bar.progress(0.95)
                                status_text.markdown("**Creating ZIP file...**")

                                # Create ZIP file
                                zip_buffer = io.BytesIO()
                                # Use ZIP_STORED (no compression) - videos are already compressed, so this is much faster
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_STORED) as zf:
                                    for root, dirs, files in os.walk(project_temp_dir):
                                        for file in files:
                                            file_path = os.path.join(root, file)
                                            arcname = os.path.join(project_name, os.path.relpath(file_path, project_temp_dir))
                                            zf.write(file_path, arcname)

                                zip_buffer.seek(0)
                                zip_data = zip_buffer.getvalue()

                                progress_bar.progress(1.0)

                                # Track stats
                                stats_tracker.increment_stat('total_clips_analyzed', len(video_files))

                                # Clean up progress indicators
                                cancel_placeholder.empty()
                                time_text.empty()
                                progress_bar.empty()
                                status_text.empty()

                                # Calculate timing
                                total_time = time.time() - download_start
                                total_fmt = f"{int(total_time // 60)}:{int(total_time % 60):02d}"
                                avg_per_clip = total_time / max(len(sorted_clips), 1)

                                # Store ZIP in session state so it persists across reruns
                                st.session_state.ready_zip_data = zip_data
                                st.session_state.ready_zip_name = f"{project_name}.zip"
                                st.session_state.ready_zip_clips = len(exported_clips)
                                st.session_state.ready_zip_xml = xml_filename
                                st.session_state.ready_zip_time = total_fmt
                                st.session_state.ready_zip_avg = avg_per_clip
                                st.session_state.auto_sort_analyzed = False
                                st.session_state.auto_sort_clips = []

                            except requests.exceptions.RequestException as e:
                                st.error(f"Download failed: {e}")
                                st.info("Make sure the Dropbox link has sharing permissions set to 'Anyone with link can view'")
                            except zipfile.BadZipFile:
                                st.error("Could not extract the downloaded file. The link may not be a valid folder share.")
                            except Exception as e:
                                if "Cancelled by user" in str(e):
                                    st.warning("Analysis cancelled. You can start a new analysis when ready.")
                                    st.session_state.analysis_cancelled = False
                                    st.session_state.auto_sort_clips = []
                                else:
                                    st.error(f"Error: {e}")

                    # Show download button if ZIP is ready (persists across reruns)
                    if st.session_state.get('ready_zip_data'):
                        st.markdown(f"""
                        <div style="background: rgba(74, 222, 128, 0.15); border: 1px solid rgba(74, 222, 128, 0.4);
                                    border-radius: 12px; padding: 20px; margin-top: 16px; margin-bottom: 16px;">
                            <div style="color: #4ade80; font-size: 18px; font-weight: 700; margin-bottom: 8px;">Ready to Download!</div>
                            <div style="color: #fff; margin-bottom: 12px;">
                                <strong>{st.session_state.ready_zip_clips}</strong> clips organized
                            </div>
                            <div style="color: #a1a1aa; font-size: 13px; margin-bottom: 8px;">
                                <strong>Contents:</strong> {st.session_state.ready_zip_clips} video files + {st.session_state.ready_zip_xml}
                            </div>
                            <div style="color: #a1a1aa; font-size: 13px;">
                                <strong>Processing time:</strong> {st.session_state.ready_zip_time} ({st.session_state.ready_zip_avg:.1f}s per clip)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.download_button(
                                label=f"Download {st.session_state.ready_zip_name}",
                                data=st.session_state.ready_zip_data,
                                file_name=st.session_state.ready_zip_name,
                                mime="application/zip",
                                type="primary",
                                use_container_width=True,
                                key="download_ready_zip"
                            )
                        with col2:
                            if st.button("Start Over", use_container_width=True):
                                st.session_state.ready_zip_data = None
                                st.session_state.ready_zip_name = None
                                st.session_state.ready_zip_clips = None
                                st.session_state.ready_zip_xml = None
                                st.session_state.ready_zip_time = None
                                st.session_state.ready_zip_avg = None
                                st.rerun()

    elif input_method == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload raw video clips",
            type=['mp4', 'mov', 'avi', 'mkv', 'mxf', 'm4v'],
            accept_multiple_files=True,
            help="Upload raw footage clips from your shoot"
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} files selected**")
            # Save to temp directory for analysis
            temp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                temp_path = os.path.join(temp_dir, uf.name)
                with open(temp_path, 'wb') as f:
                    f.write(uf.getbuffer())
                clips_to_analyze.append(temp_path)

    else:  # Local Folder Path
        folder_path = st.text_input(
            "Folder path",
            placeholder="/path/to/raw/footage/folder",
            help="Enter the full path to a folder containing raw video clips"
        )

        if folder_path and os.path.isdir(folder_path):
            # Count video files
            video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.m4v'}
            video_files = []
            for root, dirs, files in os.walk(folder_path):
                for f in files:
                    if os.path.splitext(f)[1].lower() in video_extensions:
                        video_files.append(os.path.join(root, f))

            if video_files:
                st.success(f"Found **{len(video_files)}** video clips in folder")
                clips_to_analyze = video_files
            else:
                st.warning("No video files found in this folder")
        elif folder_path:
            st.error("Folder not found. Please check the path.")

    # Analyze button
    if clips_to_analyze:
        st.markdown("---")

        # Analysis mode selection - clickable cards
        st.markdown("<div style='color: #a1a1aa; font-size: 13px; margin: 8px 0 12px 0;'>Choose analysis mode:</div>", unsafe_allow_html=True)

        # Initialize mode in session state
        if 'analysis_mode_local' not in st.session_state:
            st.session_state.analysis_mode_local = 'timeline_only'

        mode_col1, mode_col2, mode_col3 = st.columns(3)

        with mode_col1:
            is_selected = st.session_state.analysis_mode_local == 'timeline_only'
            border_color = "#4ade80" if is_selected else "#1d1d1f"
            bg_color = "rgba(74, 222, 128, 0.08)" if is_selected else "#111"
            st.markdown(f"""
            <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; padding: 20px 16px 24px 16px; text-align: center;">
                <div style="margin-bottom: 10px;">{icon('mode_timeline', 28)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 14px; margin-bottom: 8px;">Timeline Only</div>
                <div style="color: #71717a; font-size: 11px; line-height: 1.5;">Fastest (~1s/clip)<br>Room sorting + auto-structure<br>Uses full clips</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select", key="local_mode_timeline", use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.analysis_mode_local = 'timeline_only'
                st.rerun()

        with mode_col2:
            is_selected = st.session_state.analysis_mode_local == 'quick'
            border_color = "#4ade80" if is_selected else "#1d1d1f"
            bg_color = "rgba(74, 222, 128, 0.08)" if is_selected else "#111"
            st.markdown(f"""
            <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; padding: 20px 16px 24px 16px; text-align: center;">
                <div style="margin-bottom: 10px;">{icon('mode_quick', 28)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 14px; margin-bottom: 8px;">Quick Mode</div>
                <div style="color: #71717a; font-size: 11px; line-height: 1.5;">Fast (~2s/clip)<br>Adds trim detection<br>15% in/out heuristic</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select", key="local_mode_quick", use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.analysis_mode_local = 'quick'
                st.rerun()

        with mode_col3:
            is_selected = st.session_state.analysis_mode_local == 'full'
            border_color = "#4ade80" if is_selected else "#1d1d1f"
            bg_color = "rgba(74, 222, 128, 0.08)" if is_selected else "#111"
            st.markdown(f"""
            <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; padding: 20px 16px 24px 16px; text-align: center;">
                <div style="margin-bottom: 10px;">{icon('mode_full', 28)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 14px; margin-bottom: 8px;">Full Analysis</div>
                <div style="color: #71717a; font-size: 11px; line-height: 1.5;">Detailed (~15s/clip)<br>Precise trim points<br>Optical flow analysis</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select", key="local_mode_full", use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.analysis_mode_local = 'full'
                st.rerun()

        analysis_mode_local = st.session_state.analysis_mode_local
        quick_mode_local = analysis_mode_local in ["quick", "timeline_only"]
        timeline_only_local = analysis_mode_local == "timeline_only"

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            button_labels = {
                "timeline_only": "Sort Footage",
                "quick": "Analyze (Quick)",
                "full": "Analyze (Full)"
            }
            button_label = button_labels[analysis_mode_local]
            if st.button(button_label, type="primary", use_container_width=True):
                st.session_state.auto_sort_clips = []
                st.session_state.timeline_only_local = timeline_only_local
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()

                total_clips = len(clips_to_analyze)
                start_time = time.time()
                clip_times = []  # Track time per clip for better estimates

                for idx, clip_path in enumerate(clips_to_analyze):
                    clip_start = time.time()
                    filename = os.path.basename(clip_path)

                    # Calculate time estimates
                    elapsed = time.time() - start_time
                    if idx > 0:
                        avg_time_per_clip = elapsed / idx
                        remaining_clips = total_clips - idx
                        est_remaining = avg_time_per_clip * remaining_clips

                        # Format times
                        elapsed_fmt = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
                        remaining_fmt = f"{int(est_remaining // 60)}:{int(est_remaining % 60):02d}"
                        time_text.markdown(f"""
                        <div style="text-align: center; color: #a1a1aa; font-size: 12px; margin-top: 8px;">
                            Elapsed: {elapsed_fmt} &nbsp;|&nbsp; Est. remaining: {remaining_fmt}
                        </div>
                        """, unsafe_allow_html=True)

                    status_text.markdown(f"{'Sorting' if timeline_only_local else 'Analyzing'}: **{filename}** ({idx + 1}/{total_clips})")
                    progress_bar.progress((idx + 1) / total_clips)

                    # Classify room type
                    room_type, confidence = classify_clip_room(clip_path, filename)

                    # Timeline Only mode: skip quality analysis, use full clip
                    if timeline_only_local:
                        try:
                            import cv2
                            cap = cv2.VideoCapture(clip_path)
                            fps = cap.get(cv2.CAP_PROP_FPS) or 30
                            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                            duration = frame_count / fps if fps > 0 else 10
                            cap.release()
                        except:
                            duration = 10
                            fps = 30

                        quality = {
                            "duration": duration,
                            "fps": fps,
                            "best_in_point": 0,
                            "best_out_point": duration,
                            "avg_stability": 0.8,
                            "avg_exposure": 0.7,
                            "best_segment_score": 0.7,
                            "shake_start": 0,
                            "shake_end": 0,
                            "hero_duration": duration,
                            "trim_suggestion": "Full clip (Timeline Only mode)"
                        }
                    else:
                        # Analyze quality for best moments
                        quality = analyze_clip_quality(clip_path, quick_mode=quick_mode_local)

                    st.session_state.auto_sort_clips.append({
                        "path": clip_path,
                        "filename": filename,
                        "room_type": room_type,
                        "room_confidence": confidence,
                        "duration": quality.get("duration", 0),
                        "fps": quality.get("fps", 30),
                        "in_point": quality.get("best_in_point", 0),
                        "out_point": quality.get("best_out_point", quality.get("duration", 10)),
                        "stability_score": quality.get("avg_stability", 0),
                        "exposure_score": quality.get("avg_exposure", 0),
                        "segment_score": quality.get("best_segment_score", 0),
                        "shake_start": quality.get("shake_start", 0),
                        "shake_end": quality.get("shake_end", 0),
                        "hero_duration": quality.get("hero_duration", quality.get("duration", 0)),
                        "trim_suggestion": quality.get("trim_suggestion", "")
                    })

                    clip_times.append(time.time() - clip_start)

                # Show completion time
                total_time = time.time() - start_time
                total_fmt = f"{int(total_time // 60)}:{int(total_time % 60):02d}"
                time_text.markdown(f"""
                <div style="text-align: center; color: #4ade80; font-size: 12px; margin-top: 8px;">
                    Completed in {total_fmt}
                </div>
                """, unsafe_allow_html=True)
                status_text.markdown("**Analysis complete!**")

                # Track stats for Auto Sort
                stats_tracker.increment_stat('total_clips_analyzed', total_clips)

                st.session_state.auto_sort_analyzed = True
                st.rerun()

    # Display results if we have analyzed clips
    if st.session_state.auto_sort_clips:
        st.markdown("---")
        st.markdown("### Analysis Results")

        # Group by room type
        rooms_found = {}
        for clip in st.session_state.auto_sort_clips:
            room = clip["room_type"]
            if room not in rooms_found:
                rooms_found[room] = []
            rooms_found[room].append(clip)

        # Calculate totals
        total_raw_duration = sum(c["duration"] for c in st.session_state.auto_sort_clips)
        total_hero_duration = sum(c.get("hero_duration", c["duration"]) for c in st.session_state.auto_sort_clips)
        time_saved = total_raw_duration - total_hero_duration
        avg_quality = np.mean([c["stability_score"] for c in st.session_state.auto_sort_clips]) if st.session_state.auto_sort_clips else 0

        # Summary stats - 5 columns now
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="font-size: 26px; font-weight: 700; color: #7B8CDE;">{len(st.session_state.auto_sort_clips)}</div>
                <div style="font-size: 11px; color: #a1a1aa;">Total Clips</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="font-size: 26px; font-weight: 700; color: #4ade80;">{len(rooms_found)}</div>
                <div style="font-size: 11px; color: #a1a1aa;">Room Types</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="font-size: 26px; font-weight: 700; color: #fff;">{total_raw_duration:.0f}s</div>
                <div style="font-size: 11px; color: #a1a1aa;">Raw Footage</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="font-size: 26px; font-weight: 700; color: #4ade80;">{total_hero_duration:.0f}s</div>
                <div style="font-size: 11px; color: #a1a1aa;">Hero Footage</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="font-size: 26px; font-weight: 700; color: #ef4444;">{time_saved:.0f}s</div>
                <div style="font-size: 11px; color: #a1a1aa;">Shake Trimmed</div>
            </div>
            """, unsafe_allow_html=True)

        # Trim summary message
        if time_saved > 0:
            pct_saved = (time_saved / total_raw_duration * 100) if total_raw_duration > 0 else 0
            st.markdown(f"""
            <div style="background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.3);
                        border-radius: 8px; padding: 12px; margin-top: 16px; text-align: center;">
                <span style="color: #4ade80; font-weight: 600;">Auto-trimmed {time_saved:.1f}s of setup/teardown shake ({pct_saved:.0f}% of raw footage)</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Clips organized by room
        st.markdown("### Clips by Room Type")

        for room_key in ROOM_TYPES.keys():
            if room_key in rooms_found:
                room_info = ROOM_TYPES[room_key]
                clips_in_room = rooms_found[room_key]

                with st.expander(f"{room_info['name']} ({len(clips_in_room)} clips)", expanded=True):
                    for clip_idx, clip in enumerate(clips_in_room):
                        clip_key = f"{room_key}_{clip_idx}_{clip['filename']}"

                        # Get current in/out from session state or clip data
                        in_key = f"in_{clip_key}"
                        out_key = f"out_{clip_key}"

                        if in_key not in st.session_state:
                            st.session_state[in_key] = clip['in_point']
                        if out_key not in st.session_state:
                            st.session_state[out_key] = clip['out_point']

                        current_in = st.session_state[in_key]
                        current_out = st.session_state[out_key]
                        total_dur = clip['duration'] if clip['duration'] > 0 else 1

                        # Calculate trim bar percentages based on current in/out
                        in_pct = (current_in / total_dur) * 100
                        hero_pct = ((current_out - current_in) / total_dur) * 100
                        out_pct = ((total_dur - current_out) / total_dur) * 100
                        hero_dur = current_out - current_in

                        # Clip card container
                        st.markdown(f"""
                        <div style="background: #161616; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <div>
                                    <span style="color: #fff; font-weight: 600; font-size: 15px;">{clip['filename']}</span>
                                    <span style="color: #71717a; font-size: 12px; margin-left: 8px;">({clip['duration']:.1f}s total)</span>
                                </div>
                                <div style="color: #4ade80; font-size: 13px; font-weight: 600;">Hero: {hero_dur:.1f}s</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Video player and thumbnails row
                        vid_col, thumb_col = st.columns([2, 1])

                        with vid_col:
                            # Video player - starts at IN point so you see the selected portion
                            if os.path.exists(clip['path']):
                                st.video(clip['path'], start_time=int(current_in))
                            else:
                                st.markdown("""
                                <div style="background: #0a0a0a; border: 1px solid #1d1d1f; border-radius: 8px;
                                            height: 180px; display: flex; align-items: center; justify-content: center;">
                                    <span style="color: #71717a;">Video not available</span>
                                </div>
                                """, unsafe_allow_html=True)

                        with thumb_col:
                            # Thumbnails at in and out points
                            st.markdown("<div style='color: #a1a1aa; font-size: 11px; margin-bottom: 4px;'>IN / OUT Preview</div>", unsafe_allow_html=True)

                            in_thumb = extract_frame_at_timestamp(clip['path'], current_in, width=140)
                            out_thumb = extract_frame_at_timestamp(clip['path'], current_out, width=140)

                            thumb_html = '<div style="display: flex; gap: 8px;">'
                            if in_thumb:
                                thumb_html += f'''
                                <div style="text-align: center;">
                                    <img src="{in_thumb}" style="border-radius: 4px; border: 2px solid #4ade80;"/>
                                    <div style="color: #4ade80; font-size: 10px; margin-top: 2px;">IN {current_in:.1f}s</div>
                                </div>'''
                            else:
                                thumb_html += '<div style="width: 70px; height: 50px; background: #0a0a0a; border-radius: 4px;"></div>'

                            if out_thumb:
                                thumb_html += f'''
                                <div style="text-align: center;">
                                    <img src="{out_thumb}" style="border-radius: 4px; border: 2px solid #ef4444;"/>
                                    <div style="color: #ef4444; font-size: 10px; margin-top: 2px;">OUT {current_out:.1f}s</div>
                                </div>'''
                            else:
                                thumb_html += '<div style="width: 70px; height: 50px; background: #0a0a0a; border-radius: 4px;"></div>'

                            thumb_html += '</div>'
                            st.markdown(thumb_html, unsafe_allow_html=True)

                        # Unified Trim Bar - visual bar with integrated slider
                        # CSS to style the slider to match the visual bar
                        st.markdown(f"""
                        <style>
                            /* Hide default slider track and use our visual bar */
                            div[data-testid="stSlider"][data-baseweb="slider"] > div:first-child {{
                                background: transparent !important;
                            }}
                            /* Style range slider thumbs */
                            .unified-trim-container .stSlider > div > div > div[role="slider"] {{
                                background: #fff !important;
                                border: 2px solid #4ade80 !important;
                                width: 14px !important;
                                height: 14px !important;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
                            }}
                        </style>
                        <div class="unified-trim-container" style="margin: 12px 0 4px 0; position: relative;">
                            <!-- Time labels -->
                            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                <span style="color: #71717a; font-size: 11px;">0s</span>
                                <span style="color: #4ade80; font-size: 12px; font-weight: 600;">Hero: {hero_dur:.1f}s</span>
                                <span style="color: #71717a; font-size: 11px;">{total_dur:.1f}s</span>
                            </div>
                            <!-- Visual bar background showing red/green/red -->
                            <div style="height: 24px; background: #0a0a0a; border-radius: 12px; overflow: hidden; display: flex; position: relative;">
                                <div style="width: {in_pct:.1f}%; background: rgba(239, 68, 68, 0.4);"></div>
                                <div style="width: {hero_pct:.1f}%; background: linear-gradient(90deg, #16a34a, #22c55e, #4ade80, #22c55e, #16a34a);"></div>
                                <div style="width: {out_pct:.1f}%; background: rgba(239, 68, 68, 0.4);"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Slider overlaid - positioned to match the visual bar
                        trim_range = st.slider(
                            "Adjust trim points",
                            min_value=0.0,
                            max_value=float(total_dur),
                            value=(float(current_in), float(current_out)),
                            step=0.1,
                            key=f"trim_slider_{clip_key}",
                            label_visibility="collapsed",
                            format="%.1fs"
                        )

                        # Trim labels
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; margin-top: -4px; margin-bottom: 10px;">
                            <span style="color: #ef4444; font-size: 11px;">Cut: 0s - {trim_range[0]:.1f}s</span>
                            <span style="color: #ef4444; font-size: 11px;">Cut: {trim_range[1]:.1f}s - {total_dur:.1f}s</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Update if slider changed
                        slider_in, slider_out = trim_range
                        if abs(slider_in - current_in) > 0.05 or abs(slider_out - current_out) > 0.05:
                            st.session_state[in_key] = slider_in
                            st.session_state[out_key] = slider_out
                            clip['in_point'] = slider_in
                            clip['out_point'] = slider_out
                            st.rerun()

                        # Controls row - Room type and Save button
                        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 1, 1, 1.5])

                        with ctrl_col1:
                            st.markdown("<div style='color: #a1a1aa; font-size: 10px;'>ROOM TYPE</div>", unsafe_allow_html=True)

                            # Track original values for learning
                            orig_room_key = f"orig_room_{clip_key}"
                            orig_in_key = f"orig_in_{clip_key}"
                            orig_out_key = f"orig_out_{clip_key}"

                            if orig_room_key not in st.session_state:
                                st.session_state[orig_room_key] = clip['room_type']
                            if orig_in_key not in st.session_state:
                                st.session_state[orig_in_key] = clip.get('original_in', clip['in_point'])
                            if orig_out_key not in st.session_state:
                                st.session_state[orig_out_key] = clip.get('original_out', clip['out_point'])

                            new_room = st.selectbox(
                                "Room",
                                options=list(ROOM_TYPES.keys()),
                                index=list(ROOM_TYPES.keys()).index(clip['room_type']) if clip['room_type'] in ROOM_TYPES else 0,
                                key=f"room_{clip_key}",
                                label_visibility="collapsed",
                                format_func=lambda x: ROOM_TYPES.get(x, {}).get("name", x)
                            )
                            clip['room_type'] = new_room

                        with ctrl_col2:
                            st.markdown("<div style='color: #a1a1aa; font-size: 10px;'>CONFIDENCE</div>", unsafe_allow_html=True)
                            conf_color = "#4ade80" if clip['room_confidence'] > 0.7 else "#f59e0b" if clip['room_confidence'] > 0.4 else "#ef4444"
                            st.markdown(f"<div style='color: {conf_color}; font-weight: 600; padding-top: 8px;'>{clip['room_confidence']*100:.0f}%</div>", unsafe_allow_html=True)

                        with ctrl_col3:
                            st.markdown("<div style='color: #a1a1aa; font-size: 10px;'>QUALITY</div>", unsafe_allow_html=True)
                            stability_color = "#4ade80" if clip['stability_score'] > 0.7 else "#f59e0b" if clip['stability_score'] > 0.4 else "#ef4444"
                            st.markdown(f"<div style='color: {stability_color}; font-weight: 600; padding-top: 8px;'>{clip['stability_score']*100:.0f}%</div>", unsafe_allow_html=True)

                        with ctrl_col4:
                            st.markdown("<div style='color: #a1a1aa; font-size: 10px;'>SAVE CHANGES</div>", unsafe_allow_html=True)

                            # Check if anything changed
                            room_changed = new_room != st.session_state[orig_room_key]
                            trim_changed = (
                                abs(current_in - st.session_state[orig_in_key]) > 0.05 or
                                abs(current_out - st.session_state[orig_out_key]) > 0.05
                            )
                            has_changes = room_changed or trim_changed

                            save_key = f"save_{clip_key}"
                            saved_key = f"saved_{clip_key}"

                            if has_changes:
                                if st.button("💾 Save", key=save_key, type="primary", use_container_width=True):
                                    # Log room correction for learning
                                    if room_changed:
                                        stats_tracker.log_room_correction(
                                            clip['filename'],
                                            st.session_state[orig_room_key],
                                            new_room
                                        )
                                        st.session_state[orig_room_key] = new_room

                                    # Log trim adjustment for learning
                                    if trim_changed:
                                        stats_tracker.log_trim_correction(
                                            clip['filename'],
                                            st.session_state[orig_in_key],
                                            st.session_state[orig_out_key],
                                            current_in,
                                            current_out,
                                            clip['duration']
                                        )
                                        st.session_state[orig_in_key] = current_in
                                        st.session_state[orig_out_key] = current_out

                                    st.session_state[saved_key] = True
                                    st.rerun()
                            else:
                                if st.session_state.get(saved_key, False):
                                    st.markdown("<div style='color: #4ade80; font-size: 12px; padding-top: 8px;'>✓ Saved</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<div style='color: #71717a; font-size: 12px; padding-top: 8px;'>No changes</div>", unsafe_allow_html=True)

                        st.markdown("<hr style='border: none; border-top: 1px solid #1d1d1f; margin: 16px 0;'/>", unsafe_allow_html=True)

        # Handle unknown/unclassified clips
        if "unknown" in rooms_found:
            st.markdown("### Unclassified Clips")
            st.warning("These clips couldn't be automatically classified. Please assign room types manually.")
            for clip in rooms_found["unknown"]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{clip['filename']}** ({clip['duration']:.1f}s)")
                with col2:
                    new_room = st.selectbox(
                        "Assign room",
                        options=list(ROOM_TYPES.keys()),
                        key=f"unknown_room_{clip['filename']}",
                        label_visibility="collapsed",
                        format_func=lambda x: ROOM_TYPES.get(x, {}).get("name", x)
                    )
                    clip['room_type'] = new_room

        # Export section
        st.markdown("---")
        st.markdown("### Export Timeline XML")

        st.markdown("""
        <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.2);
                    border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <div style="color: #a1a1aa; font-size: 13px;">
                <strong style="color: #7B8CDE;">Relink Path:</strong> The XML will reference files at the path you specify below.
                Set this to your local Dropbox sync folder path so the NLE can find the files when you import.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            project_name = st.text_input(
                "Project name",
                value="Auto_Sort_Timeline",
                help="Name for the exported timeline"
            )

        with col2:
            export_format = st.selectbox(
                "Export format",
                ["DaVinci Resolve", "Adobe Premiere Pro", "Final Cut Pro X"],
                help="Choose your editing software"
            )

        # Relink path - where the files actually live locally
        relink_path = st.text_input(
            "Local footage folder path",
            value="/Users/Shared/Dropbox/Raw Footage",
            placeholder="/path/to/local/dropbox/folder",
            help="The local path where these clips exist (your Dropbox sync folder). The XML will use this path for file references."
        )

        # Auto-structure option
        auto_structure = st.checkbox(
            "🏠 Auto-structure timeline (standard home video flow)",
            value=True,
            help="Automatically orders clips: Establishing shot → Main rooms → Bedrooms/Baths → Backyard → Pull away"
        )

        if auto_structure:
            st.markdown("""
            <div style="background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.2);
                        border-radius: 6px; padding: 10px; margin: 8px 0; font-size: 12px; color: #a1a1aa;">
                <strong style="color: #4ade80;">Timeline order:</strong>
                Drone/Exterior → Entry → Living/Kitchen/Dining → Bedrooms → Bathrooms → Office → Backyard/Pool → Pull away
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Generate XML", type="primary", use_container_width=True):
                # Prepare clips with relink paths
                clips_for_export = []
                for clip in st.session_state.auto_sort_clips:
                    export_clip = clip.copy()
                    # Use original filename with the relink path
                    if relink_path:
                        export_clip['path'] = os.path.join(relink_path, clip['filename'])
                    clips_for_export.append(export_clip)

                # Apply home video structure ordering if enabled
                if auto_structure:
                    clips_for_export = sort_clips_for_timeline(clips_for_export, use_home_video_structure=True)

                # Generate the appropriate XML
                if export_format == "Final Cut Pro X":
                    xml_content = generate_fcpxml(clips_for_export, project_name)
                    file_ext = "fcpxml"
                else:
                    xml_content = generate_premiere_xml(clips_for_export, project_name)
                    file_ext = "xml"

                # Offer download
                st.download_button(
                    label=f"Download {project_name}.{file_ext}",
                    data=xml_content,
                    file_name=f"{project_name}.{file_ext}",
                    mime="application/xml",
                    use_container_width=True,
                    key=f"download_xml_{project_name}_{int(time.time())}"
                )

                st.success(f"XML generated with paths pointing to: `{relink_path}`")

        # =============================================
        # SAVE ORGANIZED PROJECT - Full export with renamed files
        # =============================================
        st.markdown("---")
        st.markdown("### Save Organized Project")
        st.markdown("""
        <div style="color: #a1a1aa; font-size: 13px; margin-bottom: 12px;">
            Download everything organized and ready to edit: files renamed by room type, sorted into folders, with XML included.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            save_location = st.text_input(
                "Save to folder",
                value=os.path.expanduser("~/Desktop"),
                help="Where to save the organized project folder"
            )
        with col2:
            folder_name = st.text_input(
                "Project folder name",
                value=project_name,
                help="Name for the project folder"
            )

        # Options
        col1, col2 = st.columns(2)
        with col1:
            rename_files = st.checkbox("Rename files by room type", value=True,
                help="e.g., 01_Kitchen.mp4, 02_LivingRoom.mp4")
        with col2:
            create_subfolders = st.checkbox("Organize into room folders", value=False,
                help="Create subfolders for each room type")

        # Export format for saved project
        save_export_format = st.selectbox(
            "Export format for saved project",
            ["DaVinci Resolve", "Adobe Premiere Pro", "Final Cut Pro X"],
            help="Choose your editing software",
            key="save_export_format"
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Save Organized Project", type="secondary", use_container_width=True):
                if not save_location or not os.path.isdir(save_location):
                    st.error("Please enter a valid save location")
                else:
                    try:
                        import shutil

                        # Create project folder
                        project_folder = os.path.join(save_location, folder_name)
                        os.makedirs(project_folder, exist_ok=True)

                        progress = st.progress(0)
                        status = st.empty()

                        # Check if clips have valid local paths
                        clips_to_process = st.session_state.auto_sort_clips
                        missing_files = []
                        for clip in clips_to_process:
                            if not os.path.exists(clip.get('path', '')):
                                missing_files.append(clip['filename'])

                        if missing_files and len(missing_files) == len(clips_to_process):
                            st.error("""
                            **Source files not found locally.**

                            The video files from Dropbox were temporary downloads. To save an organized project:
                            1. Use "Generate XML" above to create a timeline file
                            2. Set "Local footage folder path" to your Dropbox sync folder
                            3. Import the XML into your editing software

                            The XML will reference files in your local Dropbox folder.
                            """)
                        else:
                            # Group clips by room for ordering
                            room_order = list(ROOM_TYPES.keys())
                            sorted_clips = sorted(clips_to_process,
                                key=lambda c: (room_order.index(c['room_type']) if c['room_type'] in room_order else 99, c['filename']))

                            # Copy and rename files
                            exported_clips = []
                            room_counters = {}
                            skipped_count = 0

                            for idx, clip in enumerate(sorted_clips):
                                status.markdown(f"Copying: **{clip['filename']}** ({idx + 1}/{len(sorted_clips)})")
                                progress.progress((idx + 1) / len(sorted_clips))

                                room = clip['room_type']
                                room_name = ROOM_TYPES.get(room, {}).get('name', room).replace('/', '-').replace(' ', '')

                                # Track counter per room
                                if room not in room_counters:
                                    room_counters[room] = 1
                                else:
                                    room_counters[room] += 1

                                # Generate new filename
                                ext = os.path.splitext(clip['filename'])[1]
                                if rename_files:
                                    new_filename = f"{idx+1:02d}_{room_name}_{room_counters[room]:02d}{ext}"
                                else:
                                    new_filename = clip['filename']

                                # Determine destination
                                if create_subfolders:
                                    dest_folder = os.path.join(project_folder, room_name)
                                    os.makedirs(dest_folder, exist_ok=True)
                                    dest_path = os.path.join(dest_folder, new_filename)
                                else:
                                    dest_path = os.path.join(project_folder, new_filename)

                                # Copy file
                                try:
                                    src_path = clip['path']
                                    if os.path.exists(src_path):
                                        shutil.copy2(src_path, dest_path)

                                        # Track for XML
                                        exported_clip = clip.copy()
                                        exported_clip['path'] = dest_path
                                        exported_clip['filename'] = new_filename
                                        exported_clips.append(exported_clip)
                                    else:
                                        skipped_count += 1
                                except Exception as e:
                                    st.warning(f"Could not copy {clip['filename']}: {e}")
                                    skipped_count += 1

                            # Generate and save XML
                            status.markdown("**Generating XML...**")
                            if save_export_format == "Final Cut Pro X":
                                xml_content = generate_fcpxml(exported_clips, project_name)
                                xml_filename = f"{project_name}.fcpxml"
                            else:
                                xml_content = generate_premiere_xml(exported_clips, project_name)
                                xml_filename = f"{project_name}.xml"

                            xml_path = os.path.join(project_folder, xml_filename)
                            with open(xml_path, 'w') as f:
                                f.write(xml_content)

                            progress.progress(1.0)
                            status.markdown(f"**Done!** Saved to: `{project_folder}`")

                            if skipped_count > 0:
                                st.warning(f"Skipped {skipped_count} files that couldn't be found locally")

                            st.success(f"""
                            Project saved to **{project_folder}**

                            - {len(exported_clips)} video clips copied
                            - {xml_filename} timeline file

                            Open the folder and import the XML into your editing software.
                            """)

                    except Exception as e:
                        st.error(f"Error saving project: {str(e)}")

        # Clear results button
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        if st.button("Clear Results & Start Over"):
            st.session_state.auto_sort_clips = []
            st.session_state.auto_sort_analyzed = False
            if 'auto_sort_temp_dir' in st.session_state:
                del st.session_state.auto_sort_temp_dir
            st.rerun()

    else:
        # Show room types we detect when no clips loaded
        st.markdown("---")
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
            {icon('room_detection', 24)}
            <h3 style="color: #fff; margin: 0;">Intelligent Room Detection</h3>
        </div>
        <p style="color: #71717a; font-size: 13px; margin-bottom: 16px;">
            AI-powered recognition automatically identifies and sorts your footage into these room types
        </p>
        """, unsafe_allow_html=True)

        room_types_display = [
            ("room_drone", "Drone", "Aerial establishing shots"),
            ("room_exterior", "Exterior Front", "Front of house, curb appeal"),
            ("room_exterior_rear", "Exterior Rear", "Back of house views"),
            ("room_entry", "Entry/Foyer", "Entryway, first impressions"),
            ("room_living", "Living Room", "Main living space"),
            ("room_kitchen", "Kitchen", "Kitchen, island, appliances"),
            ("room_dining", "Dining Room", "Dining areas"),
            ("room_primary_bed", "Primary Bedroom", "Master suite"),
            ("room_bedroom", "Bedroom", "Guest rooms"),
            ("room_bathroom", "Bathroom", "Full/half bath"),
            ("room_office", "Office", "Home office, study"),
            ("room_laundry", "Laundry", "Washer, dryer, utility"),
            ("room_garage", "Garage", "Garage interior"),
            ("room_adu", "ADU", "Guest house, in-law unit"),
            ("room_backyard", "Backyard", "Yard, patio, deck"),
            ("room_pool", "Pool", "Pool, spa, outdoor"),
            ("room_yard", "Yard/Garden", "Landscaping"),
            ("room_detail", "Detail", "Close-up shots"),
        ]

        cols = st.columns(4)
        for i, (icon_key, name, desc) in enumerate(room_types_display):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 10px;
                            padding: 14px 12px; margin-bottom: 10px; text-align: center;">
                    <div style="margin-bottom: 6px;">{icon(icon_key, 22)}</div>
                    <div style="color: #fff; font-size: 12px; font-weight: 600; margin-bottom: 2px;">{name}</div>
                    <div style="color: #52525b; font-size: 10px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # Footer with stats
    render_footer()


# ============================================================================
# CALIBRATION DASHBOARD
# ============================================================================

def display_calibration_dashboard():
    """Display the calibration/learning dashboard"""

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 30px;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 8px;">
            {icon('calibration', 28)}
            <h2 style="color: #fff; margin: 0;">Calibration Dashboard</h2>
        </div>
        <p style="color: #a1a1aa; font-size: 14px;">Track feedback and improve detection accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    # Get stats from database
    stats = feedback_db.get_feedback_stats()

    # =============================================
    # SUMMARY CARDS
    # =============================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 32px; font-weight: 700; color: #7B8CDE;">{stats['total_feedback']}</div>
            <div style="font-size: 12px; color: #a1a1aa; margin-top: 4px;">Total Feedback</div>
        </div>
        """, unsafe_allow_html=True)

    # Count false positives and negatives
    total_fp = sum(check.get('false_positive', 0) for check in stats['by_check'].values())
    total_fn = sum(check.get('false_negative', 0) for check in stats['by_check'].values())
    total_correct = sum(check.get('correct', 0) for check in stats['by_check'].values())

    with col2:
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 32px; font-weight: 700; color: #4ade80;">{total_correct}</div>
            <div style="font-size: 12px; color: #a1a1aa; margin-top: 4px;">Correct</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 32px; font-weight: 700; color: #f59e0b;">{total_fp}</div>
            <div style="font-size: 12px; color: #a1a1aa; margin-top: 4px;">False Positives</div>
            <div style="font-size: 10px; color: #71717a;">Flagged but OK</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 32px; font-weight: 700; color: #ef4444;">{total_fn}</div>
            <div style="font-size: 12px; color: #a1a1aa; margin-top: 4px;">False Negatives</div>
            <div style="font-size: 10px; color: #71717a;">Missed issues</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    # =============================================
    # CHECK-BY-CHECK BREAKDOWN
    # =============================================
    if stats['by_check']:
        st.markdown("### Check Accuracy")
        st.markdown("<p style='color: #a1a1aa; font-size: 13px;'>See which checks need threshold adjustments</p>", unsafe_allow_html=True)

        for check_name, check_stats in stats['by_check'].items():
            total = sum(check_stats.values())
            correct = check_stats.get('correct', 0)
            fp = check_stats.get('false_positive', 0)
            fn = check_stats.get('false_negative', 0)

            accuracy = (correct / total * 100) if total > 0 else 0

            # Color based on accuracy
            if accuracy >= 80:
                color = "#4ade80"
                status = "Good"
            elif accuracy >= 60:
                color = "#f59e0b"
                status = "Needs tuning"
            else:
                color = "#ef4444"
                status = "Needs work"

            with st.expander(f"{check_name} — {accuracy:.0f}% accurate ({status})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Correct", correct)
                with col2:
                    st.metric("False Positives", fp, help="Flagged issues that were actually OK")
                with col3:
                    st.metric("False Negatives", fn, help="Missed issues that should have been flagged")

                # Recommendations
                if fp > fn and fp > 2:
                    st.markdown(f"""
                        <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                                    border-radius: 8px; padding: 12px; display: flex; align-items: flex-start; gap: 8px;">
                            {icon('lightbulb', 16)}
                            <span style="color: #fff; font-size: 13px;"><b>Recommendation:</b> Threshold may be too strict. Consider relaxing the {check_name} threshold to reduce false positives.</span>
                        </div>
                    """, unsafe_allow_html=True)
                elif fn > fp and fn > 2:
                    st.markdown(f"""
                        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3);
                                    border-radius: 8px; padding: 12px; display: flex; align-items: flex-start; gap: 8px;">
                            {icon('warning', 16)}
                            <span style="color: #fff; font-size: 13px;"><b>Recommendation:</b> Threshold may be too loose. Consider tightening the {check_name} threshold to catch more issues.</span>
                        </div>
                    """, unsafe_allow_html=True)
                elif total > 5 and accuracy >= 80:
                    st.markdown(f"""
                        <div style="background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.3);
                                    border-radius: 8px; padding: 12px; display: flex; align-items: center; gap: 8px;">
                            {icon('pass', 16)}
                            <span style="color: #4ade80; font-size: 13px;">This check is performing well!</span>
                        </div>
                    """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 40px; text-align: center;">
            <div style="margin-bottom: 16px;">{icon('note', 48)}</div>
            <div style="font-size: 16px; color: #fff; margin-bottom: 8px;">No feedback collected yet</div>
            <div style="font-size: 13px; color: #a1a1aa;">Run some QA checks and rate the results to start calibrating</div>
        </div>
        """, unsafe_allow_html=True)

    # =============================================
    # RECENT FEEDBACK
    # =============================================
    if stats['recent']:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        st.markdown("### Recent Feedback")

        for timestamp, filename, check_name, feedback_type in stats['recent'][:10]:
            icon_name = {"correct": "thumbs_up", "false_positive": "thumbs_down", "false_negative": "warning"}.get(feedback_type, "note")
            type_label = {"correct": "Correct", "false_positive": "False positive", "false_negative": "Missed issue"}.get(feedback_type, feedback_type)

            # Parse timestamp for display
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%b %d, %H:%M")
            except:
                time_str = timestamp[:16]

            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 8px 12px; background: #111; border-radius: 8px; margin-bottom: 6px;">
                <span style="margin-right: 12px;">{icon(icon_name, 16)}</span>
                <span style="color: #fff; font-weight: 500; flex: 1;">{check_name}</span>
                <span style="color: #a1a1aa; font-size: 12px; margin-right: 12px;">{type_label}</span>
                <span style="color: #71717a; font-size: 11px;">{time_str}</span>
            </div>
            """, unsafe_allow_html=True)

    # =============================================
    # ACTIONS
    # =============================================
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Refresh Stats", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("Clear All Feedback", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                feedback_db.clear_all_feedback()
                st.success("All feedback cleared!")
                st.session_state['confirm_clear'] = False
                st.rerun()
            else:
                st.session_state['confirm_clear'] = True
                st.warning("Click again to confirm clearing all feedback")

# ============================================================================
# FILE BROWSER MODE
# ============================================================================

def display_file_browser():
    """Display file browser for batch rating"""

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 30px;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 8px;">
            {icon('folder', 28)}
            <h2 style="color: #fff; margin: 0;">File Browser</h2>
        </div>
        <p style="color: #a1a1aa; font-size: 14px;">Browse and rate files to train the QA tool</p>
    </div>
    """, unsafe_allow_html=True)

    # Folder input
    folder_path = st.text_input(
        "Folder Path",
        placeholder="Enter path to folder with videos/photos...",
        help="Enter the full path to a folder containing files to review"
    )

    # Or Dropbox folder
    dropbox_folder = st.text_input(
        "Or Dropbox Folder Link",
        placeholder="Paste Dropbox folder share link...",
        help="Will download and extract all files from the folder"
    )

    if folder_path and os.path.isdir(folder_path):
        # Get all media files
        video_exts = {'.mp4', '.mov'}
        photo_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

        all_files = []
        for f in os.listdir(folder_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in video_exts or ext in photo_exts:
                all_files.append(f)

        all_files.sort()

        if not all_files:
            st.warning("No video or photo files found in this folder")
            return

        st.success(f"Found {len(all_files)} files")

        # File selector
        if 'current_file_idx' not in st.session_state:
            st.session_state['current_file_idx'] = 0

        # Navigation
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 3, 1, 1])

        with nav_col1:
            if st.button("← Prev", disabled=st.session_state['current_file_idx'] == 0):
                st.session_state['current_file_idx'] -= 1
                st.rerun()

        with nav_col2:
            selected_file = st.selectbox(
                "Select file",
                all_files,
                index=st.session_state['current_file_idx'],
                label_visibility="collapsed"
            )
            if all_files.index(selected_file) != st.session_state['current_file_idx']:
                st.session_state['current_file_idx'] = all_files.index(selected_file)
                st.rerun()

        with nav_col3:
            if st.button("Next →", disabled=st.session_state['current_file_idx'] >= len(all_files) - 1):
                st.session_state['current_file_idx'] += 1
                st.rerun()

        with nav_col4:
            st.caption(f"{st.session_state['current_file_idx'] + 1} / {len(all_files)}")

        # Current file info
        current_file = all_files[st.session_state['current_file_idx']]
        current_path = os.path.join(folder_path, current_file)
        ext = os.path.splitext(current_file)[1].lower()

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Display file
        col1, col2 = st.columns([2, 1])

        with col1:
            if ext in photo_exts:
                st.image(current_path, use_container_width=True)
            elif ext in video_exts:
                st.video(current_path)

        with col2:
            st.markdown(f"### {current_file}")

            file_type = "video" if ext in video_exts else "photo"

            # Quick rating buttons
            st.markdown("#### Quick Rating")

            if st.button("Good", use_container_width=True, type="primary"):
                feedback_db.add_file_rating(current_file, file_type, "good")
                st.success("Marked as good!")
                # Auto-advance
                if st.session_state['current_file_idx'] < len(all_files) - 1:
                    st.session_state['current_file_idx'] += 1
                    st.rerun()

            if st.button("Has Issues", use_container_width=True):
                feedback_db.add_file_rating(current_file, file_type, "issues")
                st.warning("Marked as having issues")

            if st.button("Bad", use_container_width=True):
                feedback_db.add_file_rating(current_file, file_type, "bad")
                st.error("Marked as bad")

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            # Notes
            note = st.text_area("Notes", placeholder="Add notes about this file...", height=100)
            if st.button("Save Note") and note:
                feedback_db.add_file_rating(current_file, file_type, "note", note)
                st.success("Note saved!")

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            # Run full QA on this file
            if st.button("Run Full QA", use_container_width=True):
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 6px;">
                            {icon('clock', 12)}
                            <span style="color: #7B8CDE; font-size: 12px;">{message}</span>
                        </div>
                    """, unsafe_allow_html=True)

                if file_type == "video":
                    report = run_video_qa(current_path, progress_callback=update_progress)
                else:
                    report = run_photo_qa(current_path, progress_callback=update_progress)

                progress_bar.empty()
                status_text.empty()

                # Use timeline report for videos, regular report for photos
                if file_type == "video":
                    display_video_review_interface(report, current_path)
                else:
                    display_report(report)

    elif dropbox_folder:
        st.markdown(f"""
            <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                        border-radius: 8px; padding: 12px; display: flex; align-items: center; gap: 8px;">
                {icon('info', 16)}
                <span style="color: #fff; font-size: 13px;">Dropbox folder browsing coming soon! For now, download the folder and enter the local path.</span>
            </div>
        """, unsafe_allow_html=True)

    else:
        # Show instructions
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 40px; text-align: center;">
            <div style="margin-bottom: 16px;">{icon('folder', 48)}</div>
            <div style="font-size: 16px; color: #fff; margin-bottom: 8px;">Enter a folder path to start browsing</div>
            <div style="font-size: 13px; color: #a1a1aa;">You can quickly rate files as good/bad to help calibrate the QA tool</div>
        </div>
        """, unsafe_allow_html=True)

        # Show file rating stats
        stats = feedback_db.get_feedback_stats()
        if stats['file_ratings']:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            st.markdown("### Your File Ratings")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Good Files", stats['file_ratings'].get('good', 0))
            with col2:
                st.metric("Files with Issues", stats['file_ratings'].get('issues', 0))
            with col3:
                st.metric("Bad Files", stats['file_ratings'].get('bad', 0))

def main():
    st.set_page_config(
        page_title="Proof by Aerial Canvas",
        page_icon="✓",
        layout="wide"
    )

    # =============================================
    # AUTHENTICATION CHECK
    # =============================================
    is_authenticated, user_info = check_authentication()

    if not is_authenticated:
        show_login_page()
        return

    # Check if team member (has access) or needs waitlist
    if not user_info.get('is_team', False):
        show_waitlist_page(user_info)
        return

    # Store current user in session for stats tracking
    if 'current_user_id' not in st.session_state:
        user = user_db.get_user_by_email(user_info.get('email', ''))
        if user:
            st.session_state.current_user_id = user['id']

    # Clean, Simple Design
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Force single black background everywhere */
    .stApp, .main, .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    .stApp > header,
    .stApp > div,
    section[data-testid="stSidebar"],
    div[data-testid="stDecoration"] {
        background: #000 !important;
        background-color: #000 !important;
    }

    * { font-family: 'Poppins', -apple-system, sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding: 0.5rem 2rem 2rem !important;
        max-width: 1000px !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5 { color: #fff !important; font-weight: 600 !important; }
    p, span, label, div { color: #a1a1aa !important; }

    /* Hero */
    .hero-section {
        text-align: center;
        padding: 20px 20px 15px;
    }
    .hero-title { font-size: 40px; color: #7B8CDE !important; }
    .hero-subtitle { color: #71717a !important; }
    .tagline {
        font-size: 15px;
        color: #a1a1aa !important;
        margin-top: 12px;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* Cards */
    .upload-card {
        background: #111;
        border: 1px solid #1d1d1f;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 8px;
    }
    .card-header { display: flex; align-items: center; gap: 12px; }
    .card-icon {
        width: 36px; height: 36px;
        background: rgba(123,140,222,0.15);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
    }
    .card-title { font-size: 15px; font-weight: 600; color: #fff !important; margin: 0; }
    .card-desc { font-size: 12px; color: #71717a !important; margin: 0; }

    /* Feature Tags - Horizontal with checkmarks */
    .feature-tags {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin-top: 14px;
    }
    .feature-tag {
        background: transparent;
        color: #a1a1aa !important;
        font-size: 12px;
        font-weight: 400;
        padding: 0;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .feature-tag::before {
        content: "✓";
        color: #7B8CDE;
        font-size: 11px;
    }

    /* Tabs - Purple pill style */
    .stTabs [data-baseweb="tab-list"] {
        background: #111 !important;
        border: 1px solid #1d1d1f !important;
        border-radius: 8px !important;
        gap: 4px !important;
        padding: 4px !important;
        width: fit-content !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #71717a !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #fff !important;
        background: rgba(255,255,255,0.05) !important;
    }
    .stTabs [aria-selected="true"] {
        background: #7B8CDE !important;
        color: #000000 !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div {
        color: #000000 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] > div {
        background: #0a0a0a !important;
        border: 1px dashed #1d1d1f !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    [data-testid="stFileUploader"] > div:hover {
        border-color: #7B8CDE !important;
    }
    [data-testid="stFileUploader"] button {
        background: #7B8CDE !important;
        color: #000 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: #9BA8E8 !important;
    }
    [data-testid="stFileUploader"] small { color: #71717a !important; }

    /* Text input */
    .stTextInput input {
        background: #111 !important;
        border: 1px solid #1d1d1f !important;
        border-radius: 8px !important;
        color: #fff !important;
        padding: 10px 14px !important;
    }
    .stTextInput input:focus {
        border-color: #7B8CDE !important;
        box-shadow: none !important;
    }
    .stTextInput input::placeholder { color: #71717a !important; }

    /* Buttons - Primary (selected/action) */
    .stButton > button[kind="primary"] {
        background: #7B8CDE !important;
        color: #000000 !important;
        border: 1px solid #7B8CDE !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
    }
    .stButton > button[kind="primary"]:hover { background: #9BA8E8 !important; border-color: #9BA8E8 !important; }
    .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span, .stButton > button[kind="primary"] div {
        color: #000000 !important;
    }

    /* Buttons - Secondary (unselected/alternative) */
    .stButton > button[kind="secondary"] {
        background: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #252525 !important;
        border-color: #7B8CDE !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="secondary"] p, .stButton > button[kind="secondary"] span, .stButton > button[kind="secondary"] div {
        color: #ffffff !important;
    }

    /* Caption/hint text - make visible */
    .stCaption, [data-testid="stCaptionContainer"], small {
        color: #a1a1aa !important;
    }

    /* Alerts */
    .stSuccess { background: rgba(74,222,128,0.1) !important; border: 1px solid rgba(74,222,128,0.3) !important; border-radius: 8px !important; }
    .stError { background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important; border-radius: 8px !important; }
    .stWarning { background: rgba(245,158,11,0.1) !important; border: 1px solid rgba(245,158,11,0.3) !important; border-radius: 8px !important; }
    .stInfo { background: rgba(123,140,222,0.1) !important; border: 1px solid rgba(123,140,222,0.3) !important; border-radius: 8px !important; }

    /* Expanders */
    .stExpander { background: #111 !important; border: 1px solid #1d1d1f !important; border-radius: 8px !important; margin-bottom: 6px !important; }

    /* Progress */
    .stProgress > div > div { background: #1d1d1f !important; }
    .stProgress > div > div > div { background: #7B8CDE !important; }

    /* Range Slider - Integrated Trim Bar Style */
    .stSlider {
        margin-top: -28px !important;  /* Overlay on visual bar */
        position: relative !important;
        z-index: 10 !important;
    }
    .stSlider > div > div > div {
        background: transparent !important;  /* Hide track - visual bar shows through */
        height: 24px !important;
    }
    .stSlider > div > div > div > div {
        background: transparent !important;  /* Hide fill - visual bar shows through */
    }
    .stSlider [data-testid="stThumbValue"] {
        color: #fff !important;
        background: #161616 !important;
        border: 1px solid #4ade80 !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-size: 11px !important;
    }
    /* Slider thumbs - white circles with green border */
    .stSlider > div > div > div > div > div {
        background: #fff !important;
        border: 3px solid #4ade80 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.5) !important;
        width: 18px !important;
        height: 18px !important;
        top: 3px !important;
    }
    .stSlider > div > div > div > div > div:hover {
        background: #4ade80 !important;
        transform: scale(1.1);
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #000; }
    ::-webkit-scrollbar-thumb { background: #1d1d1f; border-radius: 4px; }

    /* ============================================= */
    /* MODE SELECTOR - Purple Pills, Centered       */
    /* ============================================= */

    /* Center the entire radio widget */
    div[data-testid="stRadio"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }

    /* Center the entire radio component */
    div[data-testid="stRadio"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }

    div[data-testid="stRadio"] > div {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }

    /* The pill container background - centered with border */
    div[data-testid="stRadio"] > div > div[role="radiogroup"] {
        background: transparent !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 8px !important;
        display: inline-flex !important;
        justify-content: center !important;
        border: 1px solid #1d1d1f !important;
        margin: 0 auto !important;
        width: auto !important;
    }

    /* Each radio option (label) - with border */
    div[data-testid="stRadio"] label {
        background: transparent !important;
        color: #71717a !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 10px 28px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        margin: 0 !important;
        border: 1px solid #1d1d1f !important;
    }

    /* Hover state */
    div[data-testid="stRadio"] label:hover {
        color: #fff !important;
        border-color: #7B8CDE !important;
    }

    /* Selected state - PURPLE background with black text */
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #7B8CDE !important;
        color: #000 !important;
        font-weight: 600 !important;
        border-color: #7B8CDE !important;
    }

    /* Hide the actual radio circle */
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }

    /* Hide the radio circle indicator */
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* Make sure text in selected is black */
    div[data-testid="stRadio"] label[data-checked="true"] span,
    div[data-testid="stRadio"] label[data-checked="true"] p,
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #000 !important;
    }

    /* Remove any red/pink focus outlines */
    div[data-testid="stRadio"] label:focus,
    div[data-testid="stRadio"] label:focus-within,
    div[data-testid="stRadio"] input:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Override Streamlit's default red selection color */
    div[data-testid="stRadio"] label[data-checked="true"]::before,
    div[data-testid="stRadio"] label:has(input:checked)::before {
        display: none !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # =============================================
    # HEADER WITH LOGO
    # =============================================
    import base64
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proof_logo_path = os.path.join(script_dir, "Proof Logo.png")

    # Load and display logo - BIGGER
    if os.path.exists(proof_logo_path):
        with open(proof_logo_path, "rb") as f:
            proof_logo_b64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{proof_logo_b64}" style="max-width: 700px; width: 100%;">'
    else:
        logo_html = '<h1 style="color: #7B8CDE; font-size: 56px; font-weight: 600;">Proof</h1>'

    st.markdown(f"""
    <div style="text-align: center; padding: 30px 20px 20px;">
        {logo_html}
    </div>
    """, unsafe_allow_html=True)

    # =============================================
    # CUSTOM TAB BUTTONS - Photo | Video | Auto Sort
    # =============================================
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Photo"

    # CSS for custom tab buttons
    st.markdown("""
    <style>
    /* Container for tab buttons */
    .tab-container {
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
    }
    .tab-container > div {
        display: flex;
        justify-content: center;
    }
    .tab-container [data-testid="stHorizontalBlock"] {
        gap: 12px;
        padding: 8px;
        border: 1px solid #1d1d1f;
        border-radius: 12px;
        width: fit-content;
        justify-content: center;
    }
    /* Style ALL tab buttons - default state (not selected) */
    .tab-container button[kind="secondary"] {
        padding: 12px 32px !important;
        border: 1px solid #1d1d1f !important;
        border-radius: 8px !important;
        color: #71717a !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        background: transparent !important;
        min-width: 100px;
    }
    .tab-container button[kind="secondary"]:hover {
        color: #fff !important;
        border-color: #7B8CDE !important;
        background: transparent !important;
    }
    /* Selected tab - purple background with black text */
    .tab-container button[kind="primary"] {
        padding: 12px 32px !important;
        border: 1px solid #7B8CDE !important;
        border-radius: 8px !important;
        background: #7B8CDE !important;
        color: #000 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        min-width: 100px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Centered tab buttons using columns
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)

    # Calculate center columns
    spacer1, btn_col, spacer2 = st.columns([1, 2, 1])

    with btn_col:
        tab_cols = st.columns(3)
        with tab_cols[0]:
            if st.button("Photo Proof", key="btn_photo", type="primary" if st.session_state.app_mode == "Photo" else "secondary", use_container_width=True):
                st.session_state.app_mode = "Photo"
                st.rerun()
        with tab_cols[1]:
            if st.button("Video Proof", key="btn_video", type="primary" if st.session_state.app_mode == "Video" else "secondary", use_container_width=True):
                st.session_state.app_mode = "Video"
                st.rerun()
        with tab_cols[2]:
            if st.button("Auto Sort", key="btn_autosort", type="primary" if st.session_state.app_mode == "Auto Sort" else "secondary", use_container_width=True):
                st.session_state.app_mode = "Auto Sort"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    app_mode = st.session_state.app_mode

    # =============================================
    # AUTO SORT - Footage Organization & XML Export
    # =============================================
    if app_mode == "Auto Sort":
        display_auto_sort()
        return

    # =============================================
    # VIDEO PROOF
    # =============================================
    if app_mode == "Video":
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 30px;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 8px;">
                {icon('video_icon', 28)}
                <h2 style="color: #fff; margin: 0;">Video Proof</h2>
            </div>
            <p style="color: #a1a1aa; font-size: 14px;">Complete QA analysis for video deliverables - 20 automated checks</p>
        </div>
        """, unsafe_allow_html=True)

        # Feature overview cards
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 30px;">
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('technical_scan', 24)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 13px; margin-bottom: 4px;">Technical</div>
                <div style="color: #71717a; font-size: 11px;">Format, resolution, framerate</div>
            </div>
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('audio_analysis', 24)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 13px; margin-bottom: 4px;">Audio</div>
                <div style="color: #71717a; font-size: 11px;">Levels, fades, noise, music</div>
            </div>
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('color_grade', 24)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 13px; margin-bottom: 4px;">Color</div>
                <div style="color: #71717a; font-size: 11px;">Grading, consistency, log</div>
            </div>
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('edit_quality', 24)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 13px; margin-bottom: 4px;">Edit</div>
                <div style="color: #71717a; font-size: 11px;">Transitions, lower thirds</div>
            </div>
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('compliance', 24)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 13px; margin-bottom: 4px;">Compliance</div>
                <div style="color: #71717a; font-size: 11px;">Naming, deliverables</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Analysis Mode Selection
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <div style="color: #fff; font-weight: 600; font-size: 14px; margin-bottom: 12px;">Analysis Mode</div>
        </div>
        """, unsafe_allow_html=True)

        # Mode selection cards
        mode_col1, mode_col2, mode_col3 = st.columns(3)

        # Initialize session state for video analysis mode
        if 'video_analysis_mode' not in st.session_state:
            st.session_state.video_analysis_mode = 'standard'

        with mode_col1:
            quick_selected = st.session_state.video_analysis_mode == 'quick'
            if st.button(
                "Quick Scan",
                key="video_mode_quick",
                type="primary" if quick_selected else "secondary",
                use_container_width=True
            ):
                st.session_state.video_analysis_mode = 'quick'
                st.rerun()
            st.markdown("""
            <div style="text-align: center; color: #71717a; font-size: 11px; margin-top: 4px;">
                Fast • Basic checks
            </div>
            """, unsafe_allow_html=True)

        with mode_col2:
            standard_selected = st.session_state.video_analysis_mode == 'standard'
            if st.button(
                "Standard",
                key="video_mode_standard",
                type="primary" if standard_selected else "secondary",
                use_container_width=True
            ):
                st.session_state.video_analysis_mode = 'standard'
                st.rerun()
            st.markdown("""
            <div style="text-align: center; color: #71717a; font-size: 11px; margin-top: 4px;">
                Balanced • Most checks
            </div>
            """, unsafe_allow_html=True)

        with mode_col3:
            full_selected = st.session_state.video_analysis_mode == 'full'
            if st.button(
                "Full Analysis",
                key="video_mode_full",
                type="primary" if full_selected else "secondary",
                use_container_width=True
            ):
                st.session_state.video_analysis_mode = 'full'
                st.rerun()
            st.markdown("""
            <div style="text-align: center; color: #71717a; font-size: 11px; margin-top: 4px;">
                Thorough • All checks
            </div>
            """, unsafe_allow_html=True)

        # Mode description
        mode_descriptions = {
            'quick': "Technical specs, audio levels, black frames (sampled), filename validation. Skips motion & detailed color analysis.",
            'standard': "All Quick checks + full black frame scan, audio clipping/silence, color consistency. Skips motion analysis.",
            'full': "Complete analysis including motion stability, transition smoothness, and frame-by-frame quality checks."
        }

        st.markdown(f"""
        <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.2);
                    border-radius: 8px; padding: 12px; margin: 16px 0 20px;">
            <div style="color: #a1a1aa; font-size: 12px;">
                {mode_descriptions.get(st.session_state.video_analysis_mode, '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # How it Works section
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                {icon('info', 18)}
                <span style="color: #fff; font-weight: 600; font-size: 15px;">How it Works</span>
            </div>
            <ol style="color: #a1a1aa; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.8;">
                <li>Paste your Dropbox video link below</li>
                <li>We analyze technical specs, audio, color, and 20+ quality checks</li>
                <li>Frame-by-frame scanning detects black frames, exposure issues, and more</li>
                <li>Get instant pass/fail results with timestamps for any issues</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        video_tab1, video_tab2 = st.tabs(["Dropbox Link", "Upload"])

        with video_tab1:
            video_dropbox = st.text_input(
                "Dropbox Link",
                placeholder="Paste Dropbox share link...",
                key="video_dropbox",
                label_visibility="collapsed"
            )
            st.caption("Paste link and click Analyze")

            if video_dropbox:
                if st.button("Analyze", key="btn_video_dropbox", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_text = st.empty()

                    # Download phase
                    status_text.markdown("""
                    <div style="color: #7B8CDE; font-size: 13px; font-weight: 500;">
                        Downloading video from Dropbox...
                    </div>
                    """, unsafe_allow_html=True)
                    progress_bar.progress(0.05)

                    tmp_path, filename, error = download_from_dropbox(video_dropbox)

                    if error:
                        progress_bar.empty()
                        status_text.empty()
                        time_text.empty()
                        st.error(f"Error: {error}")
                    elif tmp_path:
                        progress_bar.progress(0.1)
                        status_text.markdown(f"""
                        <div style="color: #fff; font-size: 14px; font-weight: 600; margin-bottom: 4px;">
                            Analyzing: {filename[:50]}{'...' if len(filename) > 50 else ''}
                        </div>
                        """, unsafe_allow_html=True)

                        start_time = time.time()
                        estimated_total = get_total_estimated_time('video')

                        def update_video_progress_db(progress, message):
                            # Scale progress to 0.1-1.0 range (download took 0-0.1)
                            scaled_progress = 0.1 + (progress * 0.9)
                            progress_bar.progress(min(scaled_progress, 1.0))

                            elapsed = time.time() - start_time
                            elapsed_fmt = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"

                            if progress > 0:
                                remaining = (elapsed / progress) - elapsed
                                eta_fmt = f"{int(remaining // 60)}:{int(remaining % 60):02d}"
                            else:
                                eta_fmt = "calculating..."

                            time_text.markdown(f"""
                            <div style="display: flex; justify-content: center; gap: 20px; color: #a1a1aa; font-size: 12px; margin-top: 8px;">
                                <span>Elapsed: {elapsed_fmt}</span>
                                <span>ETA: {eta_fmt}</span>
                                <span style="color: #7B8CDE;">{message}</span>
                            </div>
                            """, unsafe_allow_html=True)

                        report = run_video_qa(tmp_path, progress_callback=update_video_progress_db, original_filename=filename, analysis_mode=st.session_state.video_analysis_mode)

                        progress_bar.progress(1.0)
                        elapsed = time.time() - start_time
                        elapsed_fmt = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
                        time_text.markdown(f"""
                        <div style="text-align: center; color: #4ade80; font-size: 12px; margin-top: 8px;">
                            Analysis complete in {elapsed_fmt}
                        </div>
                        """, unsafe_allow_html=True)

                        time.sleep(1)  # Brief pause to show completion
                        status_text.empty()
                        time_text.empty()
                        progress_bar.empty()

                        display_video_review_interface(report, tmp_path)
                        os.unlink(tmp_path)

        with video_tab2:
            video_file = st.file_uploader(
                "Drop video here",
                type=['mp4', 'mov'],
                key="video_upload",
                label_visibility="collapsed"
            )

            if video_file:
                suffix = Path(video_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(video_file.getvalue())
                    tmp_path = tmp.name

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()
                estimated_total = get_total_estimated_time('video')

                def update_video_progress(progress, message):
                    progress_bar.progress(progress)
                    elapsed = time.time() - start_time
                    if progress > 0:
                        remaining = (elapsed / progress) - elapsed
                    else:
                        remaining = estimated_total
                    status_text.markdown(render_progress_status(message, remaining), unsafe_allow_html=True)

                report = run_video_qa(tmp_path, progress_callback=update_video_progress, analysis_mode=st.session_state.video_analysis_mode)

                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()

                display_video_review_interface(report, tmp_path)
                os.unlink(tmp_path)

    # =============================================
    # PHOTO PROOF
    # =============================================
    if app_mode == "Photo":
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 30px;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 8px;">
                {icon('photo_icon', 28)}
                <h2 style="color: #fff; margin: 0;">Photo Proof</h2>
            </div>
            <p style="color: #a1a1aa; font-size: 14px;">Complete QA analysis for photo deliverables - singles or entire folders</p>
        </div>
        """, unsafe_allow_html=True)

        # Feature overview cards
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 30px;">
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('technical_scan', 24)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 13px; margin-bottom: 4px;">Technical</div>
                <div style="color: #71717a; font-size: 11px;">Resolution, file size, format</div>
            </div>
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('sharpness_check', 24)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 13px; margin-bottom: 4px;">Sharpness</div>
                <div style="color: #71717a; font-size: 11px;">Focus, blur detection</div>
            </div>
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('exposure_color', 24)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 13px; margin-bottom: 4px;">Exposure</div>
                <div style="color: #71717a; font-size: 11px;">Brightness, contrast, HDR</div>
            </div>
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('composition', 24)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 13px; margin-bottom: 4px;">Quality</div>
                <div style="color: #71717a; font-size: 11px;">Noise, artifacts, export</div>
            </div>
            <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="margin-bottom: 8px;">{icon('compliance', 24)}</div>
                <div style="color: #fff; font-weight: 600; font-size: 13px; margin-bottom: 4px;">Compliance</div>
                <div style="color: #71717a; font-size: 11px;">Naming, staging</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # How it Works section
        st.markdown(f"""
        <div style="background: #111; border: 1px solid #1d1d1f; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                {icon('info', 18)}
                <span style="color: #fff; font-weight: 600; font-size: 15px;">How it Works</span>
            </div>
            <ol style="color: #a1a1aa; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.8;">
                <li>Paste your Dropbox shared folder link below</li>
                <li>We analyze all photos for sharpness, exposure, color, and more</li>
                <li>Photos are also scanned for room type to help train video detection</li>
                <li>Get instant pass/fail results with previews of any issues</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        photo_tab1, photo_tab2 = st.tabs(["Dropbox Link", "Upload"])

        with photo_tab1:
            photo_dropbox = st.text_input(
                "Dropbox Link",
                placeholder="Single photo or folder link...",
                key="photo_dropbox",
                label_visibility="collapsed"
            )
            st.caption("Paste link and click Analyze · Folders will batch process")

            if photo_dropbox:
                if st.button("Analyze", key="btn_photo_dropbox", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_text = st.empty()

                    # Download phase
                    status_text.markdown("""
                    <div style="color: #7B8CDE; font-size: 13px; font-weight: 500;">
                        Downloading from Dropbox...
                    </div>
                    """, unsafe_allow_html=True)
                    progress_bar.progress(0.05)

                    tmp_path, filename, error = download_from_dropbox(photo_dropbox)

                    if error:
                        progress_bar.empty()
                        status_text.empty()
                        time_text.empty()
                        st.error(f"Error: {error}")
                    elif tmp_path:
                        # Check if it's a ZIP file (folder download)
                        is_zip = tmp_path.lower().endswith('.zip') or filename.lower().endswith('.zip')

                        if is_zip:
                            # BATCH PROCESSING
                            progress_bar.progress(0.1)
                            st.markdown(f"""
                                <div style="background: rgba(123, 140, 222, 0.1); border: 1px solid rgba(123, 140, 222, 0.3);
                                            border-radius: 8px; padding: 12px; display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                                    {icon('folder', 16)}
                                    <span style="color: #fff; font-size: 13px;">Folder detected: {filename}</span>
                                </div>
                            """, unsafe_allow_html=True)

                            status_text.markdown(f"""
                            <div style="color: #7B8CDE; font-size: 13px;">Extracting photos...</div>
                            """, unsafe_allow_html=True)
                            photo_paths = extract_zip_photos(tmp_path)

                            if not photo_paths:
                                progress_bar.empty()
                                status_text.empty()
                                time_text.empty()
                                st.error("No photos found in folder")
                            else:
                                total_photos = len(photo_paths)
                                st.success(f"Found {total_photos} photos")

                                start_time = time.time()
                                photo_times = []

                                def update_batch_progress(progress, message):
                                    # Scale to 0.15-1.0 (extraction took 0.1-0.15)
                                    scaled_progress = 0.15 + (progress * 0.85)
                                    progress_bar.progress(min(scaled_progress, 1.0))

                                    elapsed = time.time() - start_time
                                    elapsed_fmt = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"

                                    # Parse current photo from message if possible
                                    current_idx = int(progress * total_photos) if progress > 0 else 0

                                    if current_idx > 0 and elapsed > 0:
                                        avg_per_photo = elapsed / current_idx
                                        remaining_photos = total_photos - current_idx
                                        eta_seconds = avg_per_photo * remaining_photos
                                        eta_fmt = f"{int(eta_seconds // 60)}:{int(eta_seconds % 60):02d}"
                                        avg_fmt = f"{avg_per_photo:.1f}s/photo"
                                    else:
                                        eta_fmt = "calculating..."
                                        avg_fmt = "..."

                                    status_text.markdown(f"""
                                    <div style="color: #fff; font-size: 14px; font-weight: 600;">
                                        Analyzing photo {current_idx + 1} of {total_photos}
                                    </div>
                                    """, unsafe_allow_html=True)

                                    time_text.markdown(f"""
                                    <div style="display: flex; justify-content: center; gap: 20px; color: #a1a1aa; font-size: 12px; margin-top: 8px;">
                                        <span>Elapsed: {elapsed_fmt}</span>
                                        <span>ETA: {eta_fmt}</span>
                                        <span>Avg: {avg_fmt}</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                                reports = run_batch_photo_qa(photo_paths, progress_callback=update_batch_progress)

                                progress_bar.progress(1.0)
                                elapsed = time.time() - start_time
                                elapsed_fmt = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
                                time_text.markdown(f"""
                                <div style="text-align: center; color: #4ade80; font-size: 12px; margin-top: 8px;">
                                    Analyzed {total_photos} photos in {elapsed_fmt}
                                </div>
                                """, unsafe_allow_html=True)

                                time.sleep(1)
                                status_text.empty()
                                time_text.empty()
                                progress_bar.empty()

                                display_batch_report(reports)

                                # Cleanup extracted files
                                import shutil
                                extract_dir = os.path.dirname(photo_paths[0]) if photo_paths else None
                                if extract_dir:
                                    shutil.rmtree(extract_dir, ignore_errors=True)

                            os.unlink(tmp_path)
                        else:
                            # SINGLE PHOTO
                            progress_bar.progress(0.1)
                            status_text.markdown(f"""
                            <div style="color: #fff; font-size: 14px; font-weight: 600; margin-bottom: 4px;">
                                Analyzing: {filename[:50]}{'...' if len(filename) > 50 else ''}
                            </div>
                            """, unsafe_allow_html=True)

                            start_time = time.time()
                            estimated_total = get_total_estimated_time('photo')

                            def update_photo_progress_db(progress, message):
                                scaled_progress = 0.1 + (progress * 0.9)
                                progress_bar.progress(min(scaled_progress, 1.0))

                                elapsed = time.time() - start_time
                                elapsed_fmt = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"

                                if progress > 0:
                                    remaining = (elapsed / progress) - elapsed
                                    eta_fmt = f"{int(remaining // 60)}:{int(remaining % 60):02d}"
                                else:
                                    eta_fmt = "calculating..."

                                time_text.markdown(f"""
                                <div style="display: flex; justify-content: center; gap: 20px; color: #a1a1aa; font-size: 12px; margin-top: 8px;">
                                    <span>Elapsed: {elapsed_fmt}</span>
                                    <span>ETA: {eta_fmt}</span>
                                    <span style="color: #7B8CDE;">{message}</span>
                                </div>
                                """, unsafe_allow_html=True)

                            report = run_photo_qa(tmp_path, progress_callback=update_photo_progress_db, original_filename=filename)

                            progress_bar.progress(1.0)
                            elapsed = time.time() - start_time
                            elapsed_fmt = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
                            time_text.markdown(f"""
                            <div style="text-align: center; color: #4ade80; font-size: 12px; margin-top: 8px;">
                                Analysis complete in {elapsed_fmt}
                            </div>
                            """, unsafe_allow_html=True)

                            time.sleep(1)
                            status_text.empty()
                            time_text.empty()
                            progress_bar.empty()

                            display_report(report)
                            os.unlink(tmp_path)

        with photo_tab2:
            photo_file = st.file_uploader(
                "Drop photo here",
                type=['jpg', 'jpeg', 'png', 'tiff'],
                key="photo_upload",
                label_visibility="collapsed"
            )

            if photo_file:
                suffix = Path(photo_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(photo_file.getvalue())
                    tmp_path = tmp.name

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()
                estimated_total = get_total_estimated_time('photo')

                def update_photo_progress(progress, message):
                    progress_bar.progress(progress)
                    elapsed = time.time() - start_time
                    remaining = (elapsed / progress) - elapsed if progress > 0 else estimated_total
                    status_text.markdown(render_progress_status(message, remaining), unsafe_allow_html=True)

                report = run_photo_qa(tmp_path, progress_callback=update_photo_progress)

                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()

                display_report(report)
                os.unlink(tmp_path)

    # Footer with stats
    render_footer()

if __name__ == "__main__":
    main()
