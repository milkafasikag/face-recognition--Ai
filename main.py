import cv2
import pickle
import numpy as np
import preproc


with open("models/eigenface_recognizer.pkl", "rb") as f:
    eigenface_recognizer = pickle.load(f)


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


video_capture = cv2.VideoCapture(0)


def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    for x, y, w, h in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return faces


while True:
    result, video_frame = video_capture.read()

    if result is False:
        break

    flipped_frame = cv2.flip(video_frame, 1)

    faces = detect_bounding_box(video_frame)

    if len(faces):
        x, y, w, h = faces[0]

        face = video_frame[y : y + h, x : x + w]
        face = cv2.resize(face, (250, 250))
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

        preprocessed_image = preproc.preprocess_images([face])

        label = eigenface_recognizer.transform(preprocessed_image)

        cv2.putText(
            video_frame,
            "Name: " + str(label[0]),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imshow("My Face Detection Project", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
