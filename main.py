import cv2
import numpy as np
from mtcnn import MTCNN
from pathlib import Path
from PIL import Image
import dlib
def detect_face1(image):
    detector = MTCNN()
    results = detector.detect_faces(image)
    if len(results) == 0:
        try_rotated_image = np.rot90(image)
        faces = detector.detect_faces(try_rotated_image)
        if len(faces) == 0:
            return None
        else:
            x, y, w, h = faces[0]['box']
            return (x, y, w, h)
    else:
        x, y, w, h = results[0]['box']
        return (x, y, w, h)

def detect_face2(image):
    detector = MTCNN()
    image_rgb = image[:, :, ::-1]  # Convert BGR to RGB
    results = detector.detect_faces(image_rgb)
    if len(results) == 0:
        try_rotated_image = np.rot90(image)
        faces = detector.detect_faces(try_rotated_image)
        if len(faces) == 0:
            return None
        else:
            x, y, w, h = faces[0]['box']
            return (x, y, w, h)
    else:
        x, y, w, h = results[0]['box']
        return dlib.rectangle(x, y, x + w, y + h)

def display_faces(pic1, pic2):

    image1 = cv2.imread(pic1)
    image2 = cv2.imread(pic2)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face1 = detect_face1(image1)
    if face1 is None:
        print("No face found in the first image.")
        return

    face2 = detect_face1(image2)
    if face2 is None:
        print("No face found in the second image.")
        return

    # Crop the faces
    x1, y1, w1, h1 = face1
    face_img1 = image1[y1:y1+h1, x1:x1+w1]

    x2, y2, w2, h2 = face2
    face_img2 = image2[y2:y2+h2, x2:x2+w2]

    # Display cropped faces
    cv2.imshow("Face 1", face_img1)
    cv2.imshow("Face 2", face_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_faces(pic1, pic2):
    image1 = dlib.load_rgb_image(pic1)
    image2 = dlib.load_rgb_image(pic2)


    shape_predictor = dlib.shape_predictor(str(Path.cwd()) + "/Models/shape_predictor_68_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1(str(Path.cwd()) + "/Models/dlib_face_recognition_resnet_model_v1.dat")

    face1 = detect_face2(image1)
    if face1 is None:
        print("No face found in the first image.")
        return

    face2 = detect_face2(image2)
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




pic1 = str(Path.cwd()) + '/Test_Images/image4.jpg'
pic2 = str(Path.cwd()) + '/Test_Images/image3.jpg'


display_faces(pic1,pic2)
compare_faces(pic1, pic2)



