"""
User Authentication and Tracking Database for Proof by Aerial Canvas
Handles user management, waitlist, and per-user statistics.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple


class UserDatabase:
    """SQLite database for user authentication and tracking"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'users.db')
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table - stores authenticated users
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                picture_url TEXT,
                is_team_member BOOLEAN DEFAULT FALSE,
                is_waitlist BOOLEAN DEFAULT FALSE,
                first_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                login_count INTEGER DEFAULT 0
            )
        ''')

        # User stats table - aggregate stats per user
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id INTEGER PRIMARY KEY,
                total_videos_analyzed INTEGER DEFAULT 0,
                total_photos_analyzed INTEGER DEFAULT 0,
                total_clips_sorted INTEGER DEFAULT 0,
                total_issues_found INTEGER DEFAULT 0,
                total_time_saved_seconds INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Waitlist table - for non-team signups
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS waitlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                signup_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email address"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def create_user(self, email: str, name: str = None, picture_url: str = None) -> int:
        """Create a new user and return their ID"""
        is_team_member = email.lower().endswith('@aerialcanvas.com')

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (email, name, picture_url, is_team_member, last_login, login_count)
            VALUES (?, ?, ?, ?, datetime('now'), 1)
        ''', (email, name, picture_url, is_team_member))
        user_id = cursor.lastrowid

        # Initialize user stats
        cursor.execute('''
            INSERT INTO user_stats (user_id) VALUES (?)
        ''', (user_id,))

        conn.commit()
        conn.close()
        return user_id

    def update_user_login(self, email: str, name: str = None, picture_url: str = None) -> Dict:
        """Update user's last login and increment login count"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update login info
        if name and picture_url:
            cursor.execute('''
                UPDATE users
                SET last_login = datetime('now'),
                    login_count = login_count + 1,
                    name = ?,
                    picture_url = ?
                WHERE email = ?
            ''', (name, picture_url, email))
        else:
            cursor.execute('''
                UPDATE users
                SET last_login = datetime('now'),
                    login_count = login_count + 1
                WHERE email = ?
            ''', (email,))

        conn.commit()
        conn.close()

        return self.get_user_by_email(email)

    def get_or_create_user(self, email: str, name: str = None, picture_url: str = None) -> Tuple[Dict, bool]:
        """Get existing user or create new one. Returns (user_dict, is_new_user)"""
        user = self.get_user_by_email(email)
        if user:
            # Update login info
            user = self.update_user_login(email, name, picture_url)
            return user, False
        else:
            # Create new user
            user_id = self.create_user(email, name, picture_url)
            user = self.get_user_by_email(email)
            return user, True

    def is_team_member(self, email: str) -> bool:
        """Check if email is an Aerial Canvas team member"""
        return email.lower().endswith('@aerialcanvas.com')

    # Waitlist management
    def add_to_waitlist(self, email: str, name: str = None, notes: str = None) -> bool:
        """Add user to waitlist. Returns True if added, False if already exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO waitlist (email, name, notes)
                VALUES (?, ?, ?)
            ''', (email, name, notes))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False

    def is_on_waitlist(self, email: str) -> bool:
        """Check if email is on the waitlist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM waitlist WHERE email = ?', (email,))
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def get_waitlist(self) -> List[Dict]:
        """Get all waitlist entries"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM waitlist ORDER BY signup_date DESC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # User stats management
    def get_user_stats(self, user_id: int) -> Optional[Dict]:
        """Get stats for a specific user"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user_stats WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def increment_user_stat(self, user_id: int, stat_name: str, amount: int = 1):
        """Increment a user's stat by amount"""
        valid_stats = [
            'total_videos_analyzed',
            'total_photos_analyzed',
            'total_clips_sorted',
            'total_issues_found',
            'total_time_saved_seconds'
        ]
        if stat_name not in valid_stats:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f'''
            UPDATE user_stats
            SET {stat_name} = {stat_name} + ?
            WHERE user_id = ?
        ''', (amount, user_id))
        conn.commit()
        conn.close()

    def update_user_stats(self, user_id: int, videos: int = 0, photos: int = 0,
                          clips: int = 0, issues: int = 0, time_saved: int = 0):
        """Update multiple stats at once"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE user_stats
            SET total_videos_analyzed = total_videos_analyzed + ?,
                total_photos_analyzed = total_photos_analyzed + ?,
                total_clips_sorted = total_clips_sorted + ?,
                total_issues_found = total_issues_found + ?,
                total_time_saved_seconds = total_time_saved_seconds + ?
            WHERE user_id = ?
        ''', (videos, photos, clips, issues, time_saved, user_id))
        conn.commit()
        conn.close()

    # Aggregate stats
    def get_total_users(self) -> int:
        """Get total number of users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_total_team_members(self) -> int:
        """Get number of team members"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_team_member = 1')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_aggregate_stats(self) -> Dict:
        """Get aggregate stats across all users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                SUM(total_videos_analyzed) as videos,
                SUM(total_photos_analyzed) as photos,
                SUM(total_clips_sorted) as clips,
                SUM(total_issues_found) as issues,
                SUM(total_time_saved_seconds) as time_saved
            FROM user_stats
        ''')
        row = cursor.fetchone()
        conn.close()

        return {
            'total_videos': row[0] or 0,
            'total_photos': row[1] or 0,
            'total_clips': row[2] or 0,
            'total_issues': row[3] or 0,
            'total_time_saved_seconds': row[4] or 0
        }


# Global instance
user_db = UserDatabase()


class LearningDatabase:
    """SQLite database for photo classification learning/feedback"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'learning.db')
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database with learning tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Photo corrections table - stores user corrections for learning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS photo_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT NOT NULL,
                original_filename TEXT,
                predicted_room TEXT,
                corrected_room TEXT NOT NULL,
                confidence_boost REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                times_confirmed INTEGER DEFAULT 1
            )
        ''')

        # Create index for fast hash lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_image_hash ON photo_corrections(image_hash)
        ''')

        # Room detection stats - track accuracy per room type
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS room_accuracy (
                room_type TEXT PRIMARY KEY,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Video detection feedback - for log footage and other video checks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_detection_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_type TEXT NOT NULL,
                contrast REAL,
                saturation REAL,
                is_correct BOOLEAN NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def compute_image_hash(self, image_bytes: bytes) -> str:
        """Compute a perceptual hash of the image for similarity matching"""
        import hashlib
        import io

        try:
            from PIL import Image
            # Open image and resize to small standard size for hashing
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((16, 16), Image.Resampling.LANCZOS)

            # Compute average hash
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            bits = ''.join('1' if p > avg else '0' for p in pixels)

            # Convert to hex string
            hash_int = int(bits, 2)
            return format(hash_int, '064x')
        except Exception:
            # Fallback to simple MD5 hash if PIL not available or image can't be processed
            return hashlib.md5(image_bytes).hexdigest()

    def save_correction(self, image_bytes: bytes, original_filename: str,
                       predicted_room: str, corrected_room: str) -> bool:
        """Save a user's correction for future learning"""
        if predicted_room == corrected_room:
            # No correction needed, but track accuracy
            self._update_accuracy(predicted_room, correct=True)
            return False

        image_hash = self.compute_image_hash(image_bytes)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if we already have this exact image
        cursor.execute('''
            SELECT id, corrected_room, times_confirmed FROM photo_corrections
            WHERE image_hash = ?
        ''', (image_hash,))
        existing = cursor.fetchone()

        if existing:
            if existing[1] == corrected_room:
                # Same correction, increase confidence
                cursor.execute('''
                    UPDATE photo_corrections
                    SET times_confirmed = times_confirmed + 1,
                        confidence_boost = confidence_boost + 0.1
                    WHERE id = ?
                ''', (existing[0],))
            else:
                # Different correction, update it
                cursor.execute('''
                    UPDATE photo_corrections
                    SET corrected_room = ?,
                        times_confirmed = 1,
                        confidence_boost = 1.0
                    WHERE id = ?
                ''', (corrected_room, existing[0]))
        else:
            # New correction
            cursor.execute('''
                INSERT INTO photo_corrections
                (image_hash, original_filename, predicted_room, corrected_room)
                VALUES (?, ?, ?, ?)
            ''', (image_hash, original_filename, predicted_room, corrected_room))

        conn.commit()
        conn.close()

        # Track accuracy
        self._update_accuracy(predicted_room, correct=False)

        return True

    def get_learned_room(self, image_bytes: bytes) -> Optional[Tuple[str, float]]:
        """Check if we have a learned room type for this image.
        Returns (room_type, confidence) or None if not found."""
        image_hash = self.compute_image_hash(image_bytes)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT corrected_room, confidence_boost, times_confirmed
            FROM photo_corrections
            WHERE image_hash = ?
        ''', (image_hash,))
        result = cursor.fetchone()
        conn.close()

        if result:
            room_type = result[0]
            confidence = min(0.99, 0.8 + (result[1] * 0.05) + (result[2] * 0.02))
            return room_type, confidence

        return None

    def _update_accuracy(self, room_type: str, correct: bool):
        """Update accuracy tracking for a room type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO room_accuracy (room_type, total_predictions, correct_predictions)
            VALUES (?, 1, ?)
            ON CONFLICT(room_type) DO UPDATE SET
                total_predictions = total_predictions + 1,
                correct_predictions = correct_predictions + ?,
                last_updated = CURRENT_TIMESTAMP
        ''', (room_type, 1 if correct else 0, 1 if correct else 0))

        conn.commit()
        conn.close()

    def get_accuracy_stats(self) -> Dict:
        """Get accuracy statistics for all room types"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT room_type, total_predictions, correct_predictions,
                   CASE WHEN total_predictions > 0
                        THEN ROUND(correct_predictions * 100.0 / total_predictions, 1)
                        ELSE 0 END as accuracy_pct
            FROM room_accuracy
            ORDER BY total_predictions DESC
        ''')
        rows = cursor.fetchall()
        conn.close()

        return {row['room_type']: {
            'total': row['total_predictions'],
            'correct': row['correct_predictions'],
            'accuracy': row['accuracy_pct']
        } for row in rows}

    def get_total_corrections(self) -> int:
        """Get total number of corrections in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM photo_corrections')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    # Video detection feedback methods
    def save_video_detection_feedback(self, check_type: str, contrast: float,
                                       saturation: float, is_correct: bool) -> bool:
        """Save feedback on a video detection (e.g., log footage)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO video_detection_feedback (check_type, contrast, saturation, is_correct)
            VALUES (?, ?, ?, ?)
        ''', (check_type, contrast, saturation, is_correct))
        conn.commit()
        conn.close()
        return True

    def get_log_detection_thresholds(self) -> Dict:
        """Get adjusted thresholds based on user feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get average contrast/saturation for false positives
        cursor.execute('''
            SELECT AVG(contrast), AVG(saturation), COUNT(*)
            FROM video_detection_feedback
            WHERE check_type = 'log_footage' AND is_correct = 0
        ''')
        false_pos = cursor.fetchone()

        # Get average for true positives
        cursor.execute('''
            SELECT AVG(contrast), AVG(saturation), COUNT(*)
            FROM video_detection_feedback
            WHERE check_type = 'log_footage' AND is_correct = 1
        ''')
        true_pos = cursor.fetchone()

        conn.close()

        # Default thresholds
        result = {'contrast': 38, 'saturation': 30, 'feedback_count': 0}

        # Adjust if we have enough feedback
        total_feedback = (false_pos[2] or 0) + (true_pos[2] or 0)
        result['feedback_count'] = total_feedback

        if false_pos[2] and false_pos[2] >= 3 and false_pos[0]:
            # Lower thresholds if we have false positives (be more strict)
            result['contrast'] = min(result['contrast'], false_pos[0] - 5)
            result['saturation'] = min(result['saturation'], false_pos[1] - 5)

        return result

    def get_video_feedback_stats(self) -> Dict:
        """Get statistics on video detection feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT check_type,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                   SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as false_positive,
                   COUNT(*) as total
            FROM video_detection_feedback
            GROUP BY check_type
        ''')
        rows = cursor.fetchall()
        conn.close()

        return {row[0]: {
            'correct': row[1],
            'false_positive': row[2],
            'total': row[3]
        } for row in rows}


# Global learning database instance
try:
    learning_db = LearningDatabase()
except Exception as e:
    print(f"Warning: Could not initialize learning database: {e}")
    # Create a dummy learning_db that does nothing but doesn't crash
    class DummyLearningDB:
        def save_correction(self, *args, **kwargs): return False
        def get_learned_room(self, *args, **kwargs): return None
        def get_accuracy_stats(self, *args, **kwargs): return {}
        def get_total_corrections(self, *args, **kwargs): return 0
        def compute_image_hash(self, *args, **kwargs): return ""
        def save_video_detection_feedback(self, *args, **kwargs): return False
        def get_log_detection_thresholds(self, *args, **kwargs): return {'contrast': 38, 'saturation': 30, 'feedback_count': 0}
        def get_video_feedback_stats(self, *args, **kwargs): return {}
    learning_db = DummyLearningDB()
