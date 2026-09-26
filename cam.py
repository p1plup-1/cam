import os
import cv2
import time
import mediapipe as mp
import serial
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# set HEADLESS=1 in the environment to run without a display (e.g. as a systemd service on a Pi with no monitor)
HEADLESS = os.environ.get("HEADLESS") == "1"

# init camera
execution_path = os.getcwd()
camera = cv2.VideoCapture(0)

# Arduino serial connection -- change ARDUINO_PORT to match your machine (Windows: "COM3", "COM4", etc.;
# check Arduino IDE > Tools > Port, or Device Manager > Ports (COM & LPT))
ARDUINO_PORT = "COM3"
ARDUINO_BAUD = 9600

try:
    arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
    time.sleep(2)  # let the Arduino reset after the serial connection opens
    print(f"Connected to Arduino on {ARDUINO_PORT}")
except serial.SerialException:
    arduino = None
    print(f"Warning: could not open {ARDUINO_PORT} -- continuing without Arduino")

# init MediaPipe Face Landmarker (468-point face mesh)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")
landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
    )
)

mouth_drawing_spec = vision.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)

connections_style = [
    (vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS, mouth_drawing_spec),
]

# all landmark indices that make up the lips, used to position the UI label near the mouth
mouth_indices = {i for conn in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS for i in (conn.start, conn.end)}

# mouth-open ratio (inner-lip gap / eye-to-eye distance) above this counts as "open" -- raise/lower to tune sensitivity
MOUTH_OPEN_THRESHOLD = 0.1

start_ts = time.time()
last_sent_state = None  # tracks the last mouth state sent to the Arduino, so we only send on change

while True:
    # Init and FPS process
    start_time = time.time()

    # Grab a single frame of video
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    # detect facial landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_ts) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    frame_h, frame_w = frame.shape[:2]

    for face_index, face_landmarks in enumerate(result.face_landmarks):
        for connections, style in connections_style:
            vision.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=style,
            )

        # mouth-open detection: inner lip gap normalized by eye-to-eye distance (scale-invariant)
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]

        mouth_gap_px = ((upper_lip.x - lower_lip.x) * frame_w) ** 2 + ((upper_lip.y - lower_lip.y) * frame_h) ** 2
        eye_dist_px = ((left_eye.x - right_eye.x) * frame_w) ** 2 + ((left_eye.y - right_eye.y) * frame_h) ** 2
        mouth_gap_px, eye_dist_px = mouth_gap_px ** 0.5, eye_dist_px ** 0.5

        mouth_open_ratio = mouth_gap_px / eye_dist_px if eye_dist_px > 0 else 0.0
        is_mouth_open = mouth_open_ratio > MOUTH_OPEN_THRESHOLD

        label = f"Mouth: {'OPEN' if is_mouth_open else 'CLOSED'} ({mouth_open_ratio:.2f})"
        color = (0, 255, 0) if is_mouth_open else (0, 0, 255)

        mouth_top_x = int(min(face_landmarks[i].x for i in mouth_indices) * frame_w)
        mouth_top_y = int(min(face_landmarks[i].y for i in mouth_indices) * frame_h)
        cv2.putText(frame, label, (mouth_top_x, max(mouth_top_y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # send mouth state to Arduino (first face only), on change only
        if face_index == 0 and arduino is not None and is_mouth_open != last_sent_state:
            arduino.write(b'O' if is_mouth_open else b'C')
            last_sent_state = is_mouth_open

    # calculate FPS >> FPS = 1 / time to process loop
    elapsed = time.time() - start_time
    fpsInfo = "FPS: " + str(1.0 / elapsed if elapsed > 0 else 0.0)

    cv2.putText(frame, fpsInfo, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if not HEADLESS:
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' or Esc on the keyboard to quit (window must be focused -- click it first)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        # also quit if the window was closed via its X button
        if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            break

# Release handle to the webcam
camera.release()
if not HEADLESS:
    cv2.destroyAllWindows()
landmarker.close()
if arduino is not None:
    arduino.close()
