from deepface import DeepFace
import cv2
from pathlib import Path

image1 = cv2.imread(str(Path.cwd()) + '/Testimages/image1.jpg')

image2 = cv2.imread(str(Path.cwd()) + '/Testimages/image2.jpg')

image3 = cv2.imread(str(Path.cwd()) + '/Testimages/image3.jpg')

image4 = cv2.imread(str(Path.cwd()) + '/Testimages/image4.jpg')

image5 = cv2.imread(str(Path.cwd()) + '/Testimages/image5.jpg')

image6 = cv2.imread(str(Path.cwd()) + '/Testimages/image6.jpg')

result = DeepFace.verify(image3, image6)
print(result)