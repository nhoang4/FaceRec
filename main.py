import dlib
from pathlib import Path
import numpy as np
from mtcnn import MTCNN
from PIL import Image


def detect_face(image):
    detector = MTCNN()
    image_rgb = image[:, :, ::-1]  # Convert BGR to RGB
    results = detector.detect_faces(image_rgb)
    if len(results) == 0:
        try_rotated_image = Image.fromarray(np.rot90(image))
        faces = detector.detect_faces(try_rotated_image)
        if len(faces) == 0:
            return None
        else:
            x, y, w, h = faces[0]['box']
            return (x, y, w, h)
    else:
        x, y, w, h = results[0]['box']
        return dlib.rectangle(x, y, x + w, y + h)


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


image1 = dlib.load_rgb_image(str(Path.cwd()) + '/Test_Images/image3.jpg')
image2 = dlib.load_rgb_image(str(Path.cwd()) + '/Test_Images/image4.jpg')
compare_faces(image1, image2)