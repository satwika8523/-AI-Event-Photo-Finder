
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

photo_paths = [
    "dataset/event_photos/sat2.jpg",
    "dataset/event_photos/sat2_another.jpg"
]

folder = "dataset/face_embeddings"
os.makedirs(folder, exist_ok=True)

for photo_path in photo_paths:
    image = cv2.imread(photo_path)

    if image is None:
        print("Could not load:", photo_path)
        continue

    faces = app.get(image)
    print(f"\nPhoto: {photo_path}")
    print("Faces detected:", len(faces))

    photo_name = os.path.splitext(
        os.path.basename(photo_path)
    )[0]

    # IMPORTANT: This loop is inside the photo loop
    for i, face in enumerate(faces, start=1):
        output_path = os.path.join(
            folder, f"{photo_name}_face_{i}.npy"
        )
        np.save(output_path, face.embedding)
        print("Saved embedding:", output_path)

        # Crop the face
        x1, y1, x2, y2 = map(int, face.bbox)
        height, width = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped_face = image[y1:y2, x1:x2]

        # Save the face image beside its embedding
        face_image_path = os.path.join(
            folder, f"{photo_name}_face_{i}.jpg"
        )

        if cropped_face.size > 0:
            saved = cv2.imwrite(face_image_path, cropped_face)
            if saved:
                print("Saved face image:", face_image_path)
            else:
                print("Failed to save:", face_image_path)

print("\nProcessing completed!")

# from insightface.app import FaceAnalysis
# import cv2
# import numpy as np
# import os

# app = FaceAnalysis(
#     name="buffalo_l",
#     providers=["CPUExecutionProvider"]
# )
# app.prepare(ctx_id=0, det_size=(640, 640))

# photo_paths = [
#     "dataset/event_photos/sat2.jpg",
#     "dataset/event_photos/sat2_another.jpg"
# ]

# os.makedirs("dataset/face_embeddings", exist_ok=True)

# for photo_path in photo_paths:
#     image = cv2.imread(photo_path)

#     if image is None:
#         print("Could not load:", photo_path)
#         continue

#     faces = app.get(image)
#     print(f"\nPhoto: {photo_path}")
#     print("Faces detected:", len(faces))

#     photo_name = os.path.splitext(
#         os.path.basename(photo_path)
#     )[0]

# for i, face in enumerate(faces, start=1):
#     output_path = (
#         f"dataset/face_embeddings/{photo_name}_face_{i}.npy"
#     )
#     np.save(output_path, face.embedding)
#     print("Saved embedding:", output_path)

#     # Crop and save the detected face image
#     x1, y1, x2, y2 = map(int, face.bbox)
#     height, width = image.shape[:2]

#     x1 = max(0, x1)
#     y1 = max(0, y1)
#     x2 = min(width, x2)
#     y2 = min(height, y2)

#     cropped_face = image[y1:y2, x1:x2]

#     face_image_path = (
#         f"dataset/face_embeddings/{photo_name}_face_{i}.jpg"
#     )

#     if cropped_face.size > 0:
#         cv2.imwrite(face_image_path, cropped_face)
#         print("Saved face image:", face_image_path)

    # for i, face in enumerate(faces, start=1):
    #     output_path = (
    #         f"dataset/face_embeddings/{photo_name}_face_{i}.npy"
    #     )
    #     np.save(output_path, face.embedding)
    #     print("Saved:", output_path)

# print("\nProcessing completed!")

# from insightface.app import FaceAnalysis
# import cv2
# import os
# import numpy as np

# # Load the face recognition model
# app = FaceAnalysis(
#     name="buffalo_l",
#     providers=["CPUExecutionProvider"]
# )

# app.prepare(ctx_id=0, det_size=(640, 640))

# # Test image
# image_path = "dataset/event_photos/sat2.jpg"

# image = cv2.imread(image_path)

# if image is None:
#     print("Image not found:", image_path)
#     raise SystemExit

# # Detect faces and generate embeddings
# faces = app.get(image)

# print("Number of faces detected:", len(faces))

# for i, face in enumerate(faces, start=1):
#     embedding = face.embedding

#     print(f"Face {i}: embedding shape = {embedding.shape}")
#     print("First 5 values:", embedding[:5])

#     os.makedirs("dataset/face_embeddings", exist_ok=True)

#     np.save(
#         f"dataset/face_embeddings/face_{i}.npy",
#         embedding
#     )

# print("Face embeddings saved successfully!")

