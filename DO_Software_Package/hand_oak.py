# hand tracking with mediapipe (~6ms per hand)
# Camera frames are retrieved from OAK-D (70FPS)
# sending hand landmarks to OSC port 14441

import cv2
import depthai
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient  # python-osc
import numpy as np

def time_delta():
    global now_rounded, previous_timestamp
    now = cv2.getTickCount()
    time_delta = (now - previous_timestamp) / cv2.getTickFrequency() * 1000
    previous_timestamp = now
    time_delta_rounded = round(time_delta, 2)
    # print(for_what, f": {time_delta_rounded} ms")
    return time_delta_rounded

#setup udp client:
client = SimpleUDPClient("127.0.0.1", 14441)

# Create the pipeline
pipeline = depthai.Pipeline()

# Define the source
cam_rgb = pipeline.createColorCamera()
# cam_rgb.setPreviewSize(1920, 1080)  # Sensor size: 1920x1080
cam_rgb.setPreviewSize(640, 480)  # Sensor size: 1920x1080
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(60)
# cam_rgb.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)

# Define the output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

is_visible = False
fps = 0.0
previous_timestamp = cv2.getTickCount()
now_rounded = 0.0
count = 0
previous_fps = 0.0
pre = cv2.getTickCount()

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
) as hands:

    # Connect to the device and start the pipeline
    with depthai.Device(pipeline) as device:

        # Define the queues
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        # Loop over the frames
        while True:
            previous_timestamp = cv2.getTickCount()
            # Get the next frame
            in_rgb = q_rgb.tryGet()

            # If no frame available, continue
            if in_rgb is None:
                continue
            get_frame_time = time_delta()
            # Convert to cv2 format and flip
            image = in_rgb.getCvFrame()
            image = cv2.flip(image, 2)


            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #  kann sensor color order direkt in rgb??

            # inference:
            results = hands.process(image)
            image.flags.writeable = True
            inference_time = time_delta()
            # print(results)
            is_left_hand_visible = False
            is_right_hand_visible = False
            if results.multi_hand_landmarks:
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



                # Draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            client.send_message("/is_left_hand_visible", is_left_hand_visible)
            client.send_message("/is_right_hand_visible", is_right_hand_visible)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            delta_draw_landmarks = time_delta()


            # Draw black transparent rectangle
            rectangle = np.ones((image.shape[0], image.shape[1], 3))
            rectangle[:150, :270, :] = 0.5
            image = image * rectangle
            image = image.astype(np.uint8)

            current = cv2.getTickCount()
            real_loop_time = (current - pre) / cv2.getTickFrequency() * 1000
            real_fps = 1000 / real_loop_time
            real_fps = round(real_fps)
            pre = current
            # print(f"fps: {real_fps}")
            # print(f"get frame time: {get_frame_time} ms")
            # print(f"inference time: {inference_time}")
            # print("________________________")
            width = 5
            arg_list = [real_fps, width, round(real_loop_time), width, inference_time, width]
            print("\rfps: {: <{}} | loop time: {: <{}} | inference time: {: <{}}".format(
                *arg_list), end="")

            # Draw the text on the image
            fps_text = f"FPS: {real_fps:.1f}"
            loop_time_text = "Loop Time: {:.2f} ms".format(round(real_loop_time, 2))
            cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, loop_time_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "press q to QUIT", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 1, cv2.LINE_AA)

            cv2.imshow("OAK-D Camera Input", image)
            # Exit if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break

# Clean up
cv2.destroyAllWindows()