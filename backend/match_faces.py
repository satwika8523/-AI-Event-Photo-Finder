
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

folder = "dataset/face_embeddings"
photos_folder = "dataset/event_photos"

photo1_faces = sorted(
    f for f in os.listdir(folder)
    if f.startswith("sat2_face_") and f.endswith(".npy")
)

photo2_faces = sorted(
    f for f in os.listdir(folder)
    if f.startswith("sat2_another_face_") and f.endswith(".npy")
)

if not photo1_faces or not photo2_faces:
    print("Face embeddings not found. Process both photos first.")
    raise SystemExit

for file1 in photo1_faces:
    embedding1 = np.load(os.path.join(folder, file1))
    best_file = None
    best_score = -1.0

    for file2 in photo2_faces:
        embedding2 = np.load(os.path.join(folder, file2))

        score = float(cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0])

        if score > best_score:
            best_score = score
            best_file = file2

    print(f"\n{file1} -> {best_file}")
    print(f"Similarity: {best_score:.4f}")

    # Find the corresponding face images
    image1_name = file1.replace(".npy", ".jpg")
    image2_name = best_file.replace(".npy", ".jpg")
    image1_path = os.path.join(folder, image1_name)
    image2_path = os.path.join(folder, image2_name)
    # image1_path = os.path.join("dataset", "test_faces", image1_name)
    # image2_path = os.path.join("dataset", "test_faces", image2_name)

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Face image not found. Check the image filenames.")
        continue

    # Resize both faces to the same height
    height = 250
    image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
    image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))

    combined = np.hstack((image1, image2))

    cv2.imshow(
        f"Similarity: {best_score:.3f} - {file1} vs {best_file}",
        combined
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nVisual comparison completed!")


# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # Load two saved face embeddings
# face1 = np.load("dataset/face_embeddings/face_1.npy")
# face2 = np.load("dataset/face_embeddings/face_2.npy")

# # Compare the embeddings
# similarity = cosine_similarity(
#     face1.reshape(1, -1),
#     face2.reshape(1, -1)
# )[0][0]

# print("Face similarity:", round(float(similarity), 4))

# # Initial threshold for testing
# if similarity > 0.5:
#     print("These faces may belong to the same person.")
# else:
#     print("These faces may belong to different people.")
