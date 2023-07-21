import cv2
import dlib
import numpy as np
from pathlib import Path
import face_recognition
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

    faceFound = None
    for i, face in enumerate(rotated_faces):
        score = confidence_scores[i]
        if score > 0.5 and face is not None:
            faceFound  = face
            print(f"Face detected in rotation {i + 1}")
            cv2.imshow(f"Face Detected {i + 1} (Confidence: {score:.2f})", face)

        else:
            print(f"No face detected in rotation {i + 1}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return faceFound

def compare_faces(image_path1, image_path2):
    predictor = dlib.shape_predictor(str(Path.cwd()) + "/Models/shape_predictor_68_face_landmarks.dat")
    # Call rodisImage function for both images
    face_found1 = rodisImage(image_path1)
    face_found2 = rodisImage(image_path2)

    # Check if faces are detected in both images
    if face_found1 is not None and face_found2 is not None:
        # Convert the images to RGB format (face_recognition library uses RGB)
        face_found1_rgb = cv2.cvtColor(face_found1, cv2.COLOR_BGR2RGB)
        face_found2_rgb = cv2.cvtColor(face_found2, cv2.COLOR_BGR2RGB)

        # Resize the images to have the same dimensions
        desired_size = (200, 200)
        face_found1_resized = cv2.resize(face_found1_rgb, desired_size)
        face_found2_resized = cv2.resize(face_found2_rgb, desired_size)

        # Compute face embeddings for the detected faces
        face_encoding1 = face_recognition.face_encodings(face_found1_resized)[0]
        face_encoding2 = face_recognition.face_encodings(face_found2_resized)[0]

        # Calculate the Euclidean distance between the face embeddings
        euclidean_distance = face_recognition.face_distance([face_encoding1], face_encoding2)[0]

        # Draw facial landmarks on the resized faces for visualization
        landmarks1 = predictor(face_found1_resized, dlib.rectangle(0, 0, face_found1_resized.shape[1], face_found1_resized.shape[0]))
        landmarks2 = predictor(face_found2_resized, dlib.rectangle(0, 0, face_found2_resized.shape[1], face_found2_resized.shape[0]))

        for p1, p2 in zip(landmarks1.parts(), landmarks2.parts()):
            cv2.circle(face_found1_resized, (p1.x, p1.y), 1, (0, 0, 255), -1)
            cv2.circle(face_found2_resized, (p2.x, p2.y), 1, (0, 0, 255), -1)

        # Display the two detected faces with facial landmarks side by side for comparison
        comparison_image = cv2.hconcat([face_found1_resized, face_found2_resized])
        cv2.imshow("Comparison with Landmarks", comparison_image)

        print(f"Face Similarity Score: {100 - euclidean_distance * 100:.2f}%")
    else:
        print("Faces not detected in both images.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path1 = str(Path.cwd()) + '/Test_Images/image3.jpg'
image_path2 = str(Path.cwd()) + '/Test_Images/image4.jpg'
compare_faces(image_path1, image_path2)