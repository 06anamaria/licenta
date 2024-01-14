import numpy as np
import cv2
import os

if __name__ == "__main__":
    
    def create_directory(directory):

        """
        Create a directory if it does not exist
    
        Parameters:
            directory (str): The parh of the directory to be created
        """
    
        if not os.path.exists(directory):
            os.makedirs(directory)

    #Create 'images' directory if it doesn't exist

    create_directory('images')

    #Load the pre-trained face cascade classifier
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #prob here

    #Open a connection to the default camera (camera index 0)
    cam = cv2.VideoCapture(0)

    cam.set(3, 640)
    cam.set(4, 480)

    #Initiate face capture variables
    count = 0
    face_id = input("\nEnter user id (MUST be an integer) and press <return -->")
    print("\n[INFO] Initializing face capture. Look at the camera and wait...")

    while True:
        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors=5)

        for (x,y,w,h) in faces: 
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

            count +=1

            cv2.imwrite(f"./images/Users-{face_id}-{count}.jpg", gray[y:y+h, x:x+w])
            cv2.imshow('image', img)

        k=cv2.waitKey(100) % 0xff
        if k<30:
            break
        elif count >= 30:
            break

    print("\n[INFO]Succes! Exiting program")

    cam.release()
    cv2.destroyAllWindows()