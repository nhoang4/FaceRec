import cv2
import dlib
import numpy as np
from pathlib import Path

def rodisImage(image_path):
    image = cv2.imread(image_path)
    rotated_faces = []
    confidence_scores = []

    # Rotate the image in 90-degree increments
    for angle in [0, 90, 180, 270]:
        rotated_image = np.rot90(image, k=int(angle / 90))
        face_detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

        faces = face_detector(gray)

        if len(faces) > 0:
            for face in faces:
                confidence_score = face_detector.run(rotated_image, 0, 0)[1][0]
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cropped_face = rotated_image[y:y + h, x:x + w]

                # Append the cropped face and confidence score to the lists
                rotated_faces.append(cropped_face)
                confidence_scores.append(confidence_score)
        else:
            rotated_faces.append(None)
            confidence_scores.append(0)


    return rotated_faces, confidence_scores



image_path = str(Path.cwd()) + '/Test_Images/ID2.jpg'
rotated_faces, confidence_scores = rodisImage(image_path)
for i, face in enumerate(rotated_faces):


    if face is not None:
        confidence_score = confidence_scores[i]
        cv2.imshow(f"Face Detected {i+1} (Confidence: {confidence_score:.2f})", face)

    else:
        print(f"No face detected in rotation {i+1}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
