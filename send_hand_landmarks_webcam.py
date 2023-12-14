# Hand recognition with googles mediapipe lib
# Using webcam as camera input
# Sends landmarks and proportions to osc port 1337 with /landmarks and /proportions

from pythonosc.udp_client import SimpleUDPClient  # python-osc
import time
import random
import numpy as np

import cv2  # opencv-python
import mediapipe as mp  # mediapipe-silicon

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#setup udp client:
client = SimpleUDPClient("127.0.0.1", 14442)
is_visible = False
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.4
) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = cv2.flip(image, 1)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    is_left_hand_visible = False
    is_right_hand_visible = False
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            keypoints_2d = []


            handedness = handedness.classification[0].label[0:]
            # print(handedness)
            # osc_msg = [handedness, hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y]

            for datapoint in hand_landmarks.landmark:
                x = datapoint.x
                y = datapoint.y
                keypoints_2d.append(x)
                keypoints_2d.append(y)
            # print(hand_landmarks)

            if handedness == 'Left':
                is_left_hand_visible = True
                client.send_message('/left_hand_keypoints', keypoints_2d)
                # print('left')
            elif handedness == 'Right':
                is_right_hand_visible = True
                client.send_message('/right_hand_keypoints', keypoints_2d)
                # print('right')

            # print(keypoints_2d)
            # print('------------------------')

        # Draw landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    client.send_message("/is_left_hand_visible", is_left_hand_visible)
    client.send_message("/is_right_hand_visible", is_right_hand_visible)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()