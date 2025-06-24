import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from gtts import gTTS
import pygame
import time

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, left_hand, right_hand])

def run_realtime_recognition(model_path, label_encoder):
    pygame.init()
    pygame.mixer.init()
    model = load_model(model_path)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = extract_keypoints(results)
            if np.count_nonzero(keypoints) > 0:
                prediction = model.predict(keypoints.reshape(1, 1, -1))[0]
                pred_class = np.argmax(prediction)
                confidence = prediction[pred_class]
                if confidence > 0.8:
                    label = label_encoder.inverse_transform([pred_class])[0]
                    cv2.putText(image, f'{label} ({confidence:.2f})', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    tts = gTTS(text=label, lang='en')
                    tts.save("temp.mp3")
                    pygame.mixer.music.load("temp.mp3")
                    pygame.mixer.music.play()
                    time.sleep(2)

            cv2.imshow("Sign Language Recognition", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
