import cv2
import dlib
from pathlib import Path
import numpy as np
from mtcnn import MTCNN
from retinaface import RetinaFace


def detect_face(image):
    detector = MTCNN()
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    else:
        x, y, w, h = results[0]['box']
        return dlib.rectangle(x, y, x + w, y + h)


def detect_face2(image):
    detector = RetinaFace.detect_faces
    results = detector(image, model='resnet50')

    if len(results) == 0:
        return None
    else:
        # Extract the bounding box of the first face detected
        box = results[0]['box']
        x, y, w, h = box[0], box[1], box[2], box[3]
        return (x, y, w, h)


def compare_faces(image1, image2):
    shape_predictor = dlib.shape_predictor(str(Path.cwd()) + "/Models/shape_predictor_68_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1(str(Path.cwd()) + "/Models/dlib_face_recognition_resnet_model_v1.dat")

    face1 = detect_face(image1)
    if face1 is None:
        print("No face found in the first image.")
        return

    face2 = detect_face(image2)
    if face2 is None:
        print("No face found in the second image.")
        return

    shape1 = shape_predictor(image1, face1)
    face_descriptor1 = face_recognizer.compute_face_descriptor(image1, shape1)


    shape2 = shape_predictor(image2, face2)
    face_descriptor2 = face_recognizer.compute_face_descriptor(image2, shape2)


    face_descriptor1 = np.array(face_descriptor1)
    face_descriptor2 = np.array(face_descriptor2)


    distance = np.linalg.norm(face_descriptor1 - face_descriptor2)


    similarity_percentage = (1 - distance) * 100
    print("Similarity: {:.2f}%".format(similarity_percentage))





image1 = dlib.load_rgb_image(str(Path.cwd()) + '/Test_Images/ID1.jpg')
image2 = dlib.load_rgb_image(str(Path.cwd()) + '/Test_Images/ID22.jpg')
compare_faces(image1, image2)