from pathlib import Path

import streamlit as st

# Get Event ID from the shared URL
query_params = st.query_params
shared_event_id = query_params.get("event_id", "")
import cv2
import numpy as np

from insightface.app import FaceAnalysis

from backend.events import (
    create_event,
    get_event,
    get_event_folder
)

from backend.database import (
    init_db,
    save_photo_metadata,
    save_search_history,
    get_recent_photos,
    get_recent_searches
)

# Page configuration must come first
st.set_page_config(
    page_title="AI Event Photo Finder",
    page_icon="📸",
    layout="wide"
)

# Initialize database
init_db()

# Initialize gallery
if "gallery_results" not in st.session_state:
    st.session_state.gallery_results = {}
# Load face recognition model once
@st.cache_resource
def load_face_model():
    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model


# Generate normalized embedding from the largest face
def get_embedding(image, model):
    faces = model.get(image)

    if not faces:
        return None

    face = max(
        faces,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )
    )

    embedding = face.embedding

    return embedding / (
        np.linalg.norm(embedding) + 1e-10
    )


# Sidebar navigation
st.sidebar.title("📸 Event Photo Finder")


page = st.sidebar.radio(
    "Navigation",
    [
        "Create Event",
        "Find My Photos",
        "My Gallery",
        "Database Records"
    ]
)
st.title("📸 AI-Powered Smart Event Photo Finder")

# =====================================================
# CREATE EVENT PAGE
# =====================================================
if page == "Create Event":

    st.header("📸 Create a New Event")

    event_name = st.text_input(
        "Enter event name",
        placeholder="Example: College Trip 2026"
    )

    if st.button("Create Event", type="primary"):

        if not event_name.strip():
            st.warning("Please enter an event name.")

        else:
            event = create_event(event_name.strip())

            st.success("Event created successfully!")

            st.write("Event Name:", event["event_name"])
            st.write("Event ID:", event["event_id"])
            
st.divider()

st.subheader("📸 Upload Event Photos")

st.write(
    "Upload the photos belonging to this event. "
    "They will be saved in the event's folder."
)

upload_event_id = st.text_input(
    "Enter the Event ID",
    placeholder="Paste the Event ID shown above",
    key="upload_event_id"
)

uploaded_event_photos = st.file_uploader(
    "Choose event photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="event_photo_uploads"
)

if st.button("Save Event Photos", type="primary"):
    if not upload_event_id.strip():
        st.warning("Please enter the Event ID.")
    elif not uploaded_event_photos:
        st.warning("Please select at least one photo.")
    else:
        event = get_event(upload_event_id.strip())

        if event is None:
            st.error("Event not found. Please check the Event ID.")
        else:
            event_folder = get_event_folder(
                upload_event_id.strip()
            )
            photos_folder = event_folder / "photos"
            photos_folder.mkdir(parents=True, exist_ok=True)

            saved_count = 0

            for photo in uploaded_event_photos:
                safe_name = Path(photo.name).name
                photo_path = photos_folder / safe_name

                with open(photo_path, "wb") as file:
                    file.write(photo.getvalue())

                saved_count += 1

            st.success(
                f"Successfully saved {saved_count} photo(s) "
                f"for {event['event_name']}!"
            )
            # st.info(
            #     "Next, we will connect event photo uploads "
            #     "and generate a shareable event link."
            # )
# =====================================================
# PAGE 1: FIND MY PHOTOS
# =====================================================
if page == "Find My Photos":

    st.write(
        "Upload selfies and event photos to find possible matches."
    )

    st.subheader("1. Upload Selfies")

    selfie_files = st.file_uploader(
        "Select one or more selfies",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="selfies"
    )

    st.subheader("2. Select Your Event")

    event_id = st.text_input(
        "Enter Event ID",
        value=shared_event_id,
        placeholder="Example: 3552e97325",
        key="search_event_id"
    )
    
    event_files = []

    if event_id.strip():

        event = get_event(event_id.strip())

        if event is None:
            st.error("Event ID not found. Please check the ID.")

        else:
            event_folder = get_event_folder(event_id.strip())
            photos_folder = event_folder / "photos"

            if not photos_folder.exists():
                st.warning("No photos folder found for this event.")

            else:
                saved_photo_paths = []

                for photo_path in photos_folder.iterdir():
                    if photo_path.suffix.lower() in [
                        ".jpg", ".jpeg", ".png"
                    ]:
                        saved_photo_paths.append(photo_path)

                if not saved_photo_paths:
                    st.warning("No photos found in this event.")

                else:
                    st.success(
                        f"Loaded {len(saved_photo_paths)} saved event photos."
                    )

                    for photo_path in saved_photo_paths:
                        photo_bytes = photo_path.read_bytes()

                        event_files.append({
                            "name": photo_path.name,
                            "bytes": photo_bytes
                        }
                        )
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Experimental threshold; validate with known examples."
    )

    if st.button("🔍 Find Photos for Everyone", type="primary"):

        if not selfie_files:
            st.warning("Please upload at least one selfie.")

        elif not event_files:
            st.warning("Please upload event photos.")

        else:
            model = load_face_model()
            event_data = []
            seen_names = set()

            # Process event photos
            with st.spinner("Analyzing event photos..."):

                for event_file in event_files:

                    photo_bytes = event_file["bytes"]

                    photo = cv2.imdecode(
                        np.frombuffer(
                            photo_bytes,
                            dtype=np.uint8
                        ),
                        cv2.IMREAD_COLOR
                    )

                    if photo is None:
                        st.warning(
                            f"Could not read {event_file['name']}."
                        )
                        continue
    

                    # Save photo metadata
                    if event_file['name'] not in seen_names:
                        save_photo_metadata(
                            event_file['name'],
                            len(photo_bytes)
                        )
                        seen_names.add(event_file['name'])

                    faces = model.get(photo)
                    embeddings = []

                    for face in faces:
                        emb = face.embedding

                        emb = emb / (
                            np.linalg.norm(emb) + 1e-10
                        )

                        embeddings.append(emb)

                    event_data.append({
                        "name": event_file["name"],
                        "bytes": photo_bytes,
                        "embeddings": embeddings
                    })

            if not event_data:
                st.error("No valid event photos could be read.")

            else:
                # Search separately for every person
                for person_number, selfie_file in enumerate(
                    selfie_files, start=1
                ):

                    st.divider()

                    st.subheader(
                        f"Person {person_number}: {selfie_file.name}"
                    )

                    selfie = cv2.imdecode(
                        np.frombuffer(
                            selfie_file.getvalue(),
                            dtype=np.uint8
                        ),
                        cv2.IMREAD_COLOR
                    )

                    if selfie is None:
                        st.error(
                            f"Could not read {selfie_file['name']}."
                        )
                        continue

                    query_embedding = get_embedding(
                        selfie,
                        model
                    )

                    if query_embedding is None:
                        st.warning(
                            f"No face detected in {selfie_file['name']}. "
                            "Please upload a clearer selfie."
                        )
                        continue

                    matches = []

                    # Compare selfie with every event photo
                    for event in event_data:

                        if not event["embeddings"]:
                            continue

                        scores = [
                            float(
                                np.dot(
                                    query_embedding,
                                    embedding
                                )
                            )
                            for embedding in event["embeddings"]
                        ]

                        best_score = max(scores)

                        if best_score >= threshold:
                            matches.append({
                                "name": event["name"],
                                "bytes": event["bytes"],
                                "score": best_score
                            })

                    # Save search history
                    save_search_history(
                        selfie_file.name,
                        len(matches)
                    )

                    # Store results for the My Gallery page
                    st.session_state.gallery_results[
                        selfie_file.name
                    ] = matches

                    # Show immediate results
                    if matches:

                        st.success(
                            f"Found {len(matches)} possible "
                            "matching photo(s)."
                        )

                        columns = st.columns(3)

                        for i, match in enumerate(matches):

                            with columns[i % 3]:

                                st.image(
                                    match["bytes"],
                                    caption=match["name"],
                                    use_container_width=True
                                )

                                st.caption(
                                    f"Similarity: "
                                    f"{match['score']:.3f}"
                                )

                                st.download_button(
                                    "⬇️ Download Photo",
                                    data=match["bytes"],
                                    file_name=match["name"],
                                    mime="image/jpeg",
                                    key=(
                                        f"search_{person_number}_"
                                        f"{i}_{match['name']}"
                                    )
                                )

                    else:
                        st.info(
                            "No photos passed the selected "
                            "threshold for this selfie."
                        )

                st.success(
                    "Search completed! Open My Gallery "
                    "to view your saved results for this session."
                )

    st.caption(
        "Matches are estimates, not guaranteed identity "
        "verification. Validate the threshold with known examples."
    )


# =====================================================
# PAGE 2: MY GALLERY
# =====================================================
elif page == "My Gallery":

    st.header("🖼️ My Personalized Gallery")

    st.write(
        "View and download possible matching event photos "
        "from your current session."
    )

    gallery_results = st.session_state.gallery_results

    if not gallery_results:

        st.info(
            "Your gallery is empty. Go to Find My Photos "
            "and run a search first."
        )

    else:

        # Choose which person's results to display
        person_names = list(gallery_results.keys())

        selected_person = st.selectbox(
            "Select a person",
            person_names
        )

        matches = gallery_results[selected_person]

        if not matches:

            st.info(
                f"No matching photos were found for "
                f"{selected_person}."
            )

        else:

            st.success(
                f"{len(matches)} possible matching photo(s)"
            )

            columns = st.columns(3)

            for i, match in enumerate(matches):

                with columns[i % 3]:

                    st.image(
                        match["bytes"],
                        caption=match["name"],
                        use_container_width=True
                    )

                    st.caption(
                        f"Similarity: {match['score']:.3f}"
                    )

                    st.download_button(
                        "⬇️ Download Photo",
                        data=match["bytes"],
                        file_name=match["name"],
                        mime="image/jpeg",
                        key=f"gallery_{selected_person}_{i}"
                    )


# =====================================================
# PAGE 3: DATABASE RECORDS
# =====================================================
elif page == "Database Records":

    st.header("📊 Database Records")

    st.write(
        "View event photo metadata and recent search history."
    )

    photos = get_recent_photos()
    searches = get_recent_searches()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Recorded Event Photos", len(photos))

    with col2:
        st.metric("Recent Searches", len(searches))

    st.subheader("📁 Event Photo Metadata")

    if photos:

        st.dataframe(
            [
                {
                    "Filename": row[0],
                    "File Size (bytes)": row[1],
                    "Recorded Upload Time": row[2]
                }
                for row in photos
            ],
            use_container_width=True
        )

    else:
        st.info("No event photo metadata saved yet.")

    st.subheader("🔎 Recent Search History")

    if searches:

        st.dataframe(
            [
                {
                    "Selfie Filename": row[0],
                    "Possible Matches": row[1],
                    "Search Time": row[2]
                }
                for row in searches
            ],
            use_container_width=True
        )

    else:
        st.info("No search history available yet.")
