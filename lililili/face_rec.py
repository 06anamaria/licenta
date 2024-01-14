import cv2
import numpy as np
import os

if __name__ == "__main__":

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if faceCascade.empty():
        print("Failed to load cascade classifier")
        exit(1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    names = ['Ana', 'Vali']  # Add names at the correct index (index 0 is reserved for 'Ana')

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Width
    cam.set(4, 480)  # Height

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            print("Predicted ID:", id, "Confidence:", confidence)


            # Ensure the id is within the bounds of the names list
            if id < len(names) and confidence < 100:  # Adjust threshold according to your needs
                name = names[id]
                confidence_text = "  {0}%".format(round(100 - confidence))
                cv2.putText(img, str(name), (x+5, y-5), font, 1, (0, 0, 0), 2)
                cv2.putText(img, str(confidence_text), (x+5, y+h-5), font, 1, (255, 0, 0), 1)
            else:
                name = "unknown"
                confidence_text = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(name), (x+5, y-5), font, 1, (0, 0, 0), 2)
            cv2.putText(img, str(confidence_text), (x+5, y+h-5), font, 1, (255, 0, 0), 1)

        # Display the image
        cv2.imshow('camera', img)

        # Press 'ESC' for exiting video
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
