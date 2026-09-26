import multiprocessing
import os
import signal
import threading
import time
import cv2
import mediapipe as mp
import serial
import serial.tools.list_ports
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Raspberry Pi version: detect mouth open/closed and send it to the Arduino. Runs headless, with an optional preview window.
# Override settings with environment variables, e.g.  ARDUINO_PORT=/dev/ttyUSB0 CAMERA=usb python3 camPi.py

# capture resolution -- keep it small, face detection on the Pi CPU is the bottleneck. MediaPipe scales faces down to
# 256x256 internally anyway, so 320x240 loses little accuracy at arm's length; raise it if faces are far from the camera
FRAME_W = int(os.environ.get("FRAME_W", 320))
FRAME_H = int(os.environ.get("FRAME_H", 240))

# Pi camera frame rate -- higher means a fresher frame is always waiting when detection finishes
CAMERA_FPS = int(os.environ.get("CAMERA_FPS", 60))

# "auto" tries the Pi camera module (picamera2) first, then falls back to a USB webcam; force with "picam" or "usb"
CAMERA = os.environ.get("CAMERA", "auto")

# mouth-open ratio (inner-lip gap / eye-to-eye distance) above this counts as "open" -- raise/lower to tune sensitivity
MOUTH_OPEN_THRESHOLD = float(os.environ.get("MOUTH_OPEN_THRESHOLD", 0.1))
# once open, the ratio must drop below this to count as closed again -- the gap stops it flickering open/closed
# every frame when the mouth hovers right at the threshold
MOUTH_CLOSE_THRESHOLD = float(os.environ.get("MOUTH_CLOSE_THRESHOLD", MOUTH_OPEN_THRESHOLD * 0.7))

ARDUINO_BAUD = 9600

# show a live preview window -- defaults to on when a desktop is available, off when headless.
# Force with PREVIEW=1 or PREVIEW=0. Needs opencv-python (not opencv-python-headless).
PI_DESKTOP_RUNNING = os.path.exists("/tmp/.X11-unix/X0")  # the Pi's own monitor has a desktop session on display :0
PREVIEW = os.environ.get("PREVIEW", "1" if os.environ.get("DISPLAY") or PI_DESKTOP_RUNNING else "0") == "1"
if PREVIEW and not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"  # started over SSH: show the window on the Pi's monitor instead of nowhere
PREVIEW_SCALE = float(os.environ.get("PREVIEW_SCALE", 2))  # enlarge the small capture frame in the preview window


# stop cleanly on Ctrl+C or `systemctl stop` (SIGTERM) so the camera and serial port get released
running = True


def stop(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)


def open_camera():
    """Returns a read() function yielding BGR frames (or None on failure), plus a close() function."""
    if CAMERA in ("auto", "picam"):
        try:
            from picamera2 import Picamera2

            picam = Picamera2()
            # picamera2's "RGB888" is laid out B,G,R in memory -- i.e. what OpenCV calls BGR
            picam.configure(picam.create_video_configuration(
                main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
                controls={"FrameRate": CAMERA_FPS},
            ))
            picam.start()
            print("Using Pi camera module (picamera2)")
            return picam.capture_array, picam.close
        except Exception as e:
            if CAMERA == "picam":
                raise RuntimeError(f"Could not open Pi camera: {e}")
            print(f"Pi camera not available ({e}), trying USB webcam")

    cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cam.isOpened():
        raise RuntimeError("Could not open USB webcam at /dev/video0. Check it's plugged in (`ls /dev/video*`).")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the newest frame instead of a stale buffered one
    print("Using USB webcam (V4L2)")

    def read():
        ret, frame = cam.read()
        return frame if ret else None

    return read, cam.release


class FrameGrabber:
    """Reads frames on a background thread, keeping only the newest, so capture and face detection run in parallel
    and detection never works on a stale buffered frame."""

    def __init__(self, read):
        self._read = read
        self._frame = None
        self._fresh = False
        self._failed = False
        self._alive = True
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._alive:
            frame = self._read()
            with self._cond:
                if frame is None:
                    self._failed = True
                else:
                    self._frame, self._fresh = frame, True
                self._cond.notify()
            if frame is None:
                break

    def read(self):
        """Newest frame not returned yet (waits for one), or None if the camera failed or stalled."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._fresh or self._failed, timeout=2) or self._failed:
                return None
            self._fresh = False
            return self._frame

    def stop(self):
        self._alive = False
        self._thread.join(timeout=1)


def find_arduino_port():
    """ARDUINO_PORT env var wins; otherwise pick the first USB serial device (/dev/ttyACM* or /dev/ttyUSB*)."""
    if os.environ.get("ARDUINO_PORT"):
        return os.environ["ARDUINO_PORT"]
    for p in serial.tools.list_ports.comports():
        if p.device.startswith(("/dev/ttyACM", "/dev/ttyUSB")):
            return p.device
    return "/dev/ttyACM0"


def _connect_arduino(warn):
    port = find_arduino_port()
    try:
        conn = serial.Serial(port, ARDUINO_BAUD, timeout=1, write_timeout=1)
    except serial.SerialException as e:
        if warn:
            print(f"Warning: could not open Arduino on {port}: {e} (will keep retrying)", flush=True)
        return None
    time.sleep(2)  # opening the port resets the Arduino; wait for it to boot
    print(f"Connected to Arduino on {port}", flush=True)
    return conn


def _arduino_worker(wanted, wake, stop_event):
    """Runs in its own process: sends the wanted mouth state (-1 none yet, 0 closed, 1 open) whenever it changes,
    reconnecting and resending if a write fails."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process, which then stops us
    parent = os.getppid()
    conn, sent, warn = None, None, True
    while not stop_event.is_set() and os.getppid() == parent:  # also exit if the main process was killed
        if conn is None:
            conn = _connect_arduino(warn)
            if conn is None:
                warn = False
                stop_event.wait(2)
                continue
            sent, warn = None, True  # Arduino just reset, so resend the current state

        state = wanted.value
        if state < 0 or state == sent:
            wake.wait(timeout=0.5)  # nothing new to send; sleep until send() is called
            wake.clear()
            continue
        try:
            conn.write(b'O' if state else b'C')
            sent = state
        except (serial.SerialException, OSError) as e:  # includes write timeouts
            print(f"Warning: Arduino write failed ({e}), reconnecting", flush=True)
            try:
                # drop the unsent bytes first -- otherwise Linux blocks close() for up to 30s trying to deliver them
                conn.reset_output_buffer()
                conn.close()
            except Exception:
                pass
            conn = None

    if conn is not None:
        conn.close()


class ArduinoLink:
    """Sends the mouth state to the Arduino from a separate process. Serial writes kept timing out when done from a
    thread inside this busy camera/MediaPipe process, but work reliably from their own process. Reconnects on its
    own if the Arduino stalls or drops off USB. User must be in the "dialout" group: sudo usermod -aG dialout $USER

    Create it before starting the camera or MediaPipe: it forks, and forking is only safe before other threads exist."""

    def __init__(self):
        ctx = multiprocessing.get_context("fork")
        self._wanted = ctx.Value('b', -1)
        self._wake = ctx.Event()
        self._stop = ctx.Event()
        self._process = ctx.Process(target=_arduino_worker, args=(self._wanted, self._wake, self._stop), daemon=True)
        self._process.start()

    def send(self, is_mouth_open):
        self._wanted.value = 1 if is_mouth_open else 0
        self._wake.set()

    def close(self):
        self._stop.set()
        self._wake.set()
        self._process.join(timeout=3)
        if self._process.is_alive():
            self._process.terminate()


arduino = ArduinoLink()  # first, before any threads start (it forks)

camera_read, close_camera = open_camera()
grabber = FrameGrabber(camera_read)

# init MediaPipe Face Landmarker (468-point face mesh)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")
landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,  # only the first face drives the Arduino, so don't spend Pi CPU on more
    )
)

mouth_drawing_spec = vision.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)

# all landmark indices that make up the lips, used to position the preview label near the mouth
mouth_indices = {i for conn in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS for i in (conn.start, conn.end)}

start_ts = time.time()
last_sent_state = None  # tracks the last mouth state sent to the Arduino, so we only send on change
frame_count = 0
fps_window_start = time.time()

try:
    while running:
        frame = grabber.read()
        if frame is None:
            print("Failed to grab frame")
            break

        # detect facial landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_ts) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        frame_h, frame_w = frame.shape[:2]

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            # mouth-open detection: inner lip gap normalized by eye-to-eye distance (scale-invariant)
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]

            mouth_gap_px = ((upper_lip.x - lower_lip.x) * frame_w) ** 2 + ((upper_lip.y - lower_lip.y) * frame_h) ** 2
            eye_dist_px = ((left_eye.x - right_eye.x) * frame_w) ** 2 + ((left_eye.y - right_eye.y) * frame_h) ** 2
            mouth_gap_px, eye_dist_px = mouth_gap_px ** 0.5, eye_dist_px ** 0.5

            mouth_open_ratio = mouth_gap_px / eye_dist_px if eye_dist_px > 0 else 0.0
            threshold = MOUTH_CLOSE_THRESHOLD if last_sent_state else MOUTH_OPEN_THRESHOLD
            is_mouth_open = mouth_open_ratio > threshold

            # send mouth state to Arduino, on change only
            if is_mouth_open != last_sent_state:
                print(f"Mouth: {'OPEN' if is_mouth_open else 'CLOSED'} ({mouth_open_ratio:.2f})")
                arduino.send(is_mouth_open)
                last_sent_state = is_mouth_open

        if PREVIEW:
            # upscale the small capture frame for viewing; landmarks are normalized (0-1) so they draw at any size
            display = cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            display_h, display_w = display.shape[:2]

            if result.face_landmarks:
                vision.drawing_utils.draw_landmarks(
                    image=display,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mouth_drawing_spec,
                )
                label = f"Mouth: {'OPEN' if is_mouth_open else 'CLOSED'} ({mouth_open_ratio:.2f})"
                color = (0, 255, 0) if is_mouth_open else (0, 0, 255)
                mouth_top_x = int(min(face_landmarks[i].x for i in mouth_indices) * display_w)
                mouth_top_y = int(min(face_landmarks[i].y for i in mouth_indices) * display_h)
                cv2.putText(display, label, (mouth_top_x, max(mouth_top_y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('Video', display)

            # 'q' or Esc quits (window must be focused), as does closing the window
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27 or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                break

        # log FPS every 5 seconds instead of drawing it on a frame nobody sees
        frame_count += 1
        elapsed = time.time() - fps_window_start
        if elapsed >= 5:
            print(f"FPS: {frame_count / elapsed:.1f}")
            frame_count = 0
            fps_window_start = time.time()
finally:
    grabber.stop()
    close_camera()
    if PREVIEW:
        cv2.destroyAllWindows()
    landmarker.close()
    arduino.close()
    print("Stopped")
