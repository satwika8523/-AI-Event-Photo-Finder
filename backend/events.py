
import json
import uuid
from pathlib import Path
from datetime import datetime

# Main event storage folder
# EVENTS_DIR = Path("event_storage")
# EVENTS_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR = Path(__file__).resolve().parent.parent / "event_storage_new"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
def create_event(event_name):
    """Create a new event with a unique ID."""

    event_id = uuid.uuid4().hex[:10]

    event_folder = EVENTS_DIR / event_id
    event_folder.mkdir(parents=True, exist_ok=True)

    event_info = {
        "event_id": event_id,
        "event_name": event_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    info_path = event_folder / "event.json"

    with open(info_path, "w", encoding="utf-8") as file:
        json.dump(event_info, file, indent=4)

    return event_info
def get_event(event_id):
    """Retrieve event information using its ID."""

    info_path = EVENTS_DIR / event_id / "event.json"

    if not info_path.exists():
        return None

    with open(info_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_event_folder(event_id):
    """Return the folder path for an event."""
    return EVENTS_DIR / event_id

# def get_event(event_id):
#     """Retrieve event information using its ID."""

#     info_path = EVENTS_DIR / event_id / "event.json"

#     if not info_path.exists():
#         return None

#     with open(info_path, "r", encoding="utf-8") as file:
#         return json.load(file)
