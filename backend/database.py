
import sqlite3
from datetime import datetime
import os

DB_PATH = "database/event_photo_finder.db"


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                file_size INTEGER,
                uploaded_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                selfie_name TEXT,
                match_count INTEGER,
                searched_at TEXT
            )
        """)


def save_photo_metadata(filename, file_size):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO event_photos
                (filename, file_size, uploaded_at)
            VALUES (?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                file_size = excluded.file_size
        """, (
            filename,
            file_size,
            datetime.now().isoformat(timespec="seconds")
        ))


def save_search_history(selfie_name, match_count):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO search_history
                (selfie_name, match_count, searched_at)
            VALUES (?, ?, ?)
        """, (
            selfie_name,
            match_count,
            datetime.now().isoformat(timespec="seconds")
        ))


def get_recent_photos():
    with get_connection() as conn:
        return conn.execute("""
            SELECT filename, file_size, uploaded_at
            FROM event_photos
            ORDER BY id DESC
        """).fetchall()


def get_recent_searches():
    with get_connection() as conn:
        return conn.execute("""
            SELECT selfie_name, match_count, searched_at
            FROM search_history
            ORDER BY id DESC
            LIMIT 20
        """).fetchall()
