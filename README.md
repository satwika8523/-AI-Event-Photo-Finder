# 📸 AI-Powered Smart Event Photo Finder

An AI-based web application that helps users find their personal photos from large event photo collections using face recognition technology.

## 1. Project Overview

Events such as college functions, weddings, birthday celebrations, cultural programs, and conferences generate hundreds or thousands of photographs. Finding a particular person's photos from these large collections can be difficult and time-consuming.

The **AI-Powered Smart Event Photo Finder** is designed to solve this problem using Artificial Intelligence and Computer Vision.

The application allows event organizers to upload event photographs and create a unique Event ID. Participants can enter the Event ID and upload their selfie. The system analyzes the uploaded selfie, identifies the face, compares it with faces in the event photographs, and displays the matching photos.

This reduces manual photo searching and provides users with a simple and convenient way to find their event memories.

## 2. Problem Statement

During large events, photographers capture a huge number of photographs containing different people. After the event, participants often need to search through hundreds of images to find their own photographs.

Traditional methods require users to manually check each photograph, which consumes time and effort. Sharing all photographs with everyone also makes it difficult to locate specific pictures.

Therefore, an automated system is needed to identify and retrieve photographs based on a person's face.

## 3. Proposed Solution

The proposed system uses face detection and face recognition techniques to automatically identify matching photographs.

The organizer uploads event photographs and receives a unique Event ID. Participants use this Event ID to access the event and upload their selfies.

The system generates numerical facial representations called embeddings and compares them with the embeddings generated from event photographs. Based on similarity scores, the system retrieves the matching images.

The application provides a personalized photo gallery containing the photographs in which the participant's face is detected and matched.

## 4. Objectives

The main objectives of this project are:

- To develop an AI-based event photo searching application.
- To reduce manual searching of photographs.
- To detect human faces in event photographs.
- To extract facial features using face recognition technology.
- To compare facial embeddings using cosine similarity.
- To retrieve photographs that match the participant's selfie.
- To organize photographs based on individual events.
- To provide a simple and user-friendly interface.
- To make event photo searching faster and more convenient.

## 5. Key Features

### 5.1 Event Creation
- Organizers can create a new event.
- Each event receives a unique Event ID.
- Event details are stored for future access.
- Separate folders are created for individual events.

### 5.2 Event Photo Upload
- Organizers can upload multiple event photographs.
- Uploaded photographs are associated with the corresponding event.
- Photo metadata can be stored for management and retrieval.

### 5.3 Face Detection
- The system detects human faces in uploaded photographs.
- OpenCV and the YuNet face detection model are used.
- Detected faces are located using bounding boxes.

### 5.4 Face Recognition
- InsightFace is used to extract facial features.
- The extracted facial features are converted into numerical embeddings.
- Each embedding represents important facial characteristics.

### 5.5 Face Matching
- The participant uploads a selfie.
- The system extracts the facial embedding from the selfie.
- It compares the selfie embedding with stored event-photo embeddings.
- Cosine similarity is used to measure the similarity between facial embeddings.

### 5.6 Personalized Photo Gallery
- Matching photographs are displayed to the participant.
- Users can view their matched event photographs.
- Matching photographs can be downloaded through the application.

### 5.7 Database Management
- SQLite is used to store photo metadata and search history.
- Event information is maintained separately using JSON files.
- This helps organize event-related information.

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| OpenCV | Image processing and face detection |
| YuNet | Face detection model |
| InsightFace | Facial feature extraction |
| Buffalo_L | Pretrained face analysis model |
| NumPy | Numerical operations and embedding comparison |
| SQLite | Storing photo metadata and search history |
| JSON | Storing event information |
| ONNX Runtime | Running AI models |
| Git and GitHub | Source code management |

## 7. System Architecture

The application follows a simple workflow involving event creation, photograph uploading, facial feature extraction, face matching, and result generation.

### Workflow

1. **Event Creation:** The organizer creates an event and receives a unique Event ID.
2. **Photo Upload:** The organizer uploads photographs related to that event.
3. **Face Detection:** The system detects faces in the uploaded photographs.
4. **Feature Extraction:** InsightFace extracts facial features and generates embeddings.
5. **Selfie Upload:** The participant enters the Event ID and uploads a selfie.
6. **Embedding Generation:** The system generates an embedding for the participant's face.
7. **Face Matching:** The system compares the selfie embedding with event-photo embeddings using cosine similarity.
8. **Result Generation:** Matching photographs are displayed in a personalized gallery.

### Architecture Diagram

```text
           ORGANIZER
               |
               v
          Create Event
               |
               v
          Generate Event ID
               |
               v
         Upload Event Photos
               |
               v
          Face Detection
               |
               v
        Face Feature Extraction
               |
               v
         Store Embeddings
               |
               v
       PARTICIPANT UPLOADS
              SELFIE
               |
               v
          Face Detection
               |
               v
        Generate Embedding
               |
               v
       Compare Embeddings
        (Cosine Similarity)
               |
               v
        Matching Photos
               |
               v
       Personalized Gallery
```

## 8. Working Methodology

### Step 1: Event Creation

The organizer enters the event name. The application generates a unique Event ID using UUID-based identification.

A separate folder is created for storing the event's information and related photographs.

### Step 2: Uploading Photographs

The organizer uploads event photographs. These photographs are associated with the selected Event ID.

### Step 3: Face Detection

The application uses OpenCV with the YuNet model to locate faces in the photographs.

Face detection identifies the location of faces in an image and helps extract individual face regions.

### Step 4: Facial Feature Extraction

InsightFace processes the detected faces and extracts facial features.

These features are represented as numerical vectors called **face embeddings**.

The Buffalo_L model generates 512-dimensional face embeddings.

### Step 5: Selfie Processing

The participant uploads a selfie after entering the correct Event ID.

The application detects the participant's face and generates its embedding using InsightFace.

### Step 6: Similarity Calculation

The system compares the participant's embedding with the embeddings extracted from event photographs.

Cosine similarity is used to measure how similar two embedding vectors are.

A higher similarity score indicates greater similarity between the facial representations.

### Step 7: Displaying Results

The application identifies photographs that satisfy the configured matching threshold and displays them to the participant.

The participant can view and download the matching photographs.

## 9. Face Matching Algorithm

This project uses cosine similarity to compare facial embeddings.

Cosine similarity measures the cosine of the angle between two vectors.

**Formula:**

```text
Cosine Similarity = (A · B) / (||A|| × ||B||)
```

Where:

- A = Embedding of the uploaded selfie.
- B = Embedding of a face from an event photograph.
- A · B = Dot product of the two vectors.
- ||A|| and ||B|| = Magnitudes of the vectors.

The similarity score is used to determine whether two facial representations are sufficiently similar.

The matching threshold is configurable and should be tested with real event photographs to balance correct matches and incorrect matches.

## 10. Project Structure

```text
AI_Event_Photo_Finder/
│
├── backend/
│   ├── database.py
│   ├── events.py
│   ├── face_detection.py
│   ├── face_recognition.py
│   └── match_faces.py
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── database/
│
├── models/
│
├── dataset/
│   ├── event_photos/
│   ├── test_faces/
│   └── face_embeddings/
│
└── event_storage_new/
```

**Folder Description:**

- `backend/`: Contains the main backend modules.
- `database.py`: Handles SQLite database operations and photo metadata.
- `events.py`: Handles event creation and event information.
- `face_detection.py`: Detects faces in photographs.
- `face_recognition.py`: Extracts facial embeddings using InsightFace.
- `match_faces.py`: Compares facial embeddings to identify matching faces.
- `app.py`: Contains the Streamlit application interface.
- `requirements.txt`: Contains the Python package dependencies.
- `models/`: Stores the face detection model files.
- `dataset/`: Used for local testing and storing sample images and embeddings.
- `database/`: Stores the SQLite database.
- `event_storage_new/`: Stores event-specific information and uploaded event data.

*Note: Some runtime folders are generated automatically and may not be included in the GitHub repository.*

## 11. Installation and Execution

Follow these steps to run the application locally.

### Prerequisites

Install the following software:

- Python
- Visual Studio Code
- Git
- Internet connection for installing dependencies and obtaining required model files.

### Step 1: Clone the Repository

```bash
git clone https://github.com/satwika8523/-AI-Event-Photo-Finder.git
```

### Step 2: Open the Project Folder

```bash
cd -AI-Event-Photo-Finder
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```bash
venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 12. Applications

This project can be useful in several real-world situations:

- College functions and cultural events.
- Weddings and family celebrations.
- Birthday parties.
- Corporate events.
- Conferences and seminars.
- Sports events.
- School and university functions.

## 13. Advantages

- Reduces manual photo searching.
- Saves time and effort.
- Provides personalized photo results.
- Supports event-wise photo organization.
- Uses AI-based face recognition.
- Provides a simple web-based interface.
- Makes it easier for participants to find their photographs.

## 14. Limitations

- Face matching accuracy depends on image quality and lighting conditions.
- Blurred images and partially visible faces may affect recognition.
- Large event collections require efficient storage and processing.
- Similar-looking faces may produce incorrect matches.
- A suitable similarity threshold needs testing and adjustment.
- Cloud deployment requires appropriate persistent storage and model configuration.

## 15. Future Enhancements

The following improvements can be considered in future versions:

- Integration with cloud storage such as AWS S3 or Cloudinary.
- Support for large-scale event photo collections.
- Improved face recognition accuracy.
- Advanced user authentication and event access control.
- Sharing personalized photo galleries through secure links.
- Search and filtering based on participant names.
- Improved mobile-friendly interface.
- Better processing performance for large events.
- Automatic background processing of uploaded photographs.

## 16. Privacy and Security

Facial photographs and face embeddings are sensitive personal data. The application should process them only with appropriate consent.

Future deployment should include secure event access, restricted photo sharing, appropriate data retention policies, and protection of stored facial information.

Event organizers should ensure that participants understand how their photographs and facial data are used.

## 17. Conclusion

The **AI-Powered Smart Event Photo Finder** demonstrates how Artificial Intelligence and Computer Vision can simplify the process of searching for personal photographs from large event collections.

By combining face detection, facial feature extraction, and cosine similarity, the application helps participants find photographs containing their faces without manually checking every image.

The project provides a practical application of Python, OpenCV, InsightFace, NumPy, SQLite, and Streamlit. It can be further improved through cloud integration, enhanced security, and scalable image processing.

## 18. Developer

**Project Title:** AI-Powered Smart Event Photo Finder

**Project Type:** Individual Project

**Domain:** Artificial Intelligence and Computer Vision

**Developed By:** Yegireddy Hema Satwika

**GitHub Repository:**  
https://github.com/satwika8523/-AI-Event-Photo-Finder

-
