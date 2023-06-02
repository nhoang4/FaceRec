import cv2
import dlib
from pathlib import Path

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.cnn_face_detection_model_v1(str(Path.cwd()) + "/mmod_human_face_detector.dat")
    faces = detector(gray)
    return faces


def compare_faces(image1, image2):
    face_recognizer = dlib.face_recognition_model_v1(str(Path.cwd()) + "/dlib_face_recognition_resnet_model_v1.dat")
    face1 = detect_faces(image1)
    if len(face1) == 0:
        print("No faces found in the first image.")
        return


    face2 = detect_faces(image2)
    if len(face2) == 0:
        print("No faces found in the second image.")
        return

    shape1 = face_recognizer.compute_face_descriptor(image1, face1)

    shape2 = face_recognizer.compute_face_descriptor(image2, face2)

    distance = dlib.distance(shape1, shape2)


    if distance < 0.6:
        print("The faces are similar.")
    else:
        print("The faces are not similar.")


# Load the images
image1 = cv2.imread(str(Path.cwd()) + '/Testimages/ID1.jpg')
image2 = cv2.imread(str(Path.cwd()) + '/Testimages/ID1.jpg')

# Compare the faces in the two images
compare_faces(image1, image2)