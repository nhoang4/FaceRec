import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.preprocessing import Normalizer
from pathlib import Path


# Load FaceNet model
model = load_model(str(Path.cwd())+'/facenet_keras.h5')

# Normalize face
normalizer = Normalizer(norm='l2')

def extract_face_features(img):
    # Convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect faces in the image
    face_cascade = cv2.CascadeClassifier(str(Path.cwd()) + '/haarcascade_frontalface_default.xml')
    #to be determined of the scales
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        # Extract the face region from the image
        face = img[y:y+h, x:x+w]
        # Resize the face image to the input shape required by FaceNet model
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32')
        # Preprocess the face image
        face = preprocess_input(face)
        return face
    else:
        return None



def get_face_embeddings(face):
    face = np.expand_dims(face, axis=0)
    embeddings = model.predict(face)
    embeddings = normalizer.transform(embeddings)
    return embeddings

def compare_faces(embeddings1, embeddings2):
    distance = np.linalg.norm(embeddings1 - embeddings2)
    similarity = 1 / (1 + distance)
    return similarity





image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')
face1 = extract_face_features(image1)
face2 = extract_face_features(image2)
if face1 is not None and face2 is not None:
    fe1 = get_face_embeddings(face1)
    fe2 = get_face_embeddings(face2)
    num = compare_faces(fe, fe)
    print('Similarity:', num)
else:
    print('No faces detected in one or both of the images.')