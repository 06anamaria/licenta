import cv2
import numpy as np
from PIL import Image
import os

def getImagesAndLabels(path):
    """
    Load face images and corresponding labels from the given directory path.

    Parameters: 
        path(str): Directory path containing face images

    Returns: 
        list: List of face samples
        list: list of corresponding labels
    """
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        print("Processing:", imagePath)
        filename = os.path.split(imagePath)[-1]

        # Splitting by '-' and taking the second element (User ID)
        parts = filename.split('-')
        if len(parts) < 3 or not parts[1].isdigit():
            print(f"Skipping file due to unexpected filename format: {imagePath}")
            continue

        id = int(parts[1])  # Extracting the User ID as an integer

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

# Rest of your main code remains the same


if __name__ == "__main__":

    # Directory path where the images are stored
    path = './images/'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("\n[INFO] Training...")

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    faces, ids = getImagesAndLabels(path)

    # Train the recognizer with the face samples and corresponding labels
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')

    print("\n[INFO] {0} faces trained. Exiting program".format(len(np.unique(ids))))
