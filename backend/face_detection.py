import cv2

# -----------------------------
# 1. Load YuNet face detector
# -----------------------------

model_path = "models/face_detection_yunet_2023mar.onnx"

detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

# -----------------------------
# 2. Path to test image
# -----------------------------

image_path = "dataset/event_photos/sat2.jpg"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load the image.")
    print("Check this path:")
    print(image_path)
    exit()

print("Image loaded successfully!")

# -----------------------------
# 3. Set image size
# -----------------------------

height, width = image.shape[:2]

detector.setInputSize((width, height))


# -----------------------------
# 4. Detect faces
# -----------------------------

_, faces = detector.detect(image)

if faces is None:
    print("Number of faces detected: 0")

else:
    print("Number of faces detected:", len(faces))

    # Create folder to save cropped faces
    import os
    os.makedirs("dataset/test_faces", exist_ok=True)

    # Save each detected face
    for i, face in enumerate(faces):
        x, y, w, h = face[:4].astype(int)

        cropped_face = image[y:y+h, x:x+w]

        filename = f"dataset/test_faces/face_{i+1}.jpg"
        cv2.imwrite(filename, cropped_face)

        print("Saved:", filename)

    # Draw boxes
    for face in faces:
        x, y, w, h = face[:4].astype(int)
        confidence = face[-1]

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            image,
            f"{confidence:.2f}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

# -----------------------------
# 5. Display result
# -----------------------------

cv2.imshow("YuNet Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# # -----------------------------
# # 4. Detect faces
# # -----------------------------

# _, faces = detector.detect(image)

# if faces is None:
#     print("Number of faces detected: 0")
# else:
#     print("Number of faces detected:", len(faces))

#     # Draw boxes
#     for face in faces:

#         x, y, w, h = face[:4].astype(int)

#         confidence = face[-1]

#         cv2.rectangle(
#             image,
#             (x, y),
#             (x + w, y + h),
#             (255, 0, 0),
#             2
#         )

#         cv2.putText(
#             image,
#             f"{confidence:.2f}",
#             (x, y - 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 255, 0),
#             1
#         )

# # -----------------------------
# # 5. Display result
# # -----------------------------

# cv2.imshow("YuNet Face Detection", image)

# cv2.waitKey(0)

# cv2.destroyAllWindows()