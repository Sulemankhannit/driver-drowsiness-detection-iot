# ------------------ Import Required Libraries ------------------
import cv2
from scipy.spatial import distance
import RPi.GPIO as GPIO
from imutils import face_utils
import numpy as np
import time
import dlib
from picamera2 import Picamera2
from pushover import Client as PushoverClient
from twilio.rest import Client as TwilioClient
import face_recognition
import os

# ------------------ Cloud Service Credentials ------------------
PUSHOVER_USER_KEY = "YOUR_PUSHOVER_USER_KEY"
PUSHOVER_API_TOKEN = "YOUR_PUSHOVER_API_TOKEN"
TWILIO_ACCOUNT_SID = "YOUR_TWILIO_SID"
TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"
TWILIO_PHONE_NUMBER = "YOUR_TWILIO_NUMBER"
EMERGENCY_CONTACT_NUMBER = "YOUR_EMERGENCY_NUMBER"

# ------------------ Initialize Cloud Clients ------------------
pushover_client = PushoverClient(PUSHOVER_USER_KEY, api_token=PUSHOVER_API_TOKEN)
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# ------------------ Cloud Alert Functions ------------------
def send_phone_alarm(message, title="Driver Alert!", priority=1):
    """Send push notification to mobile using Pushover app."""
    print(f"Sending Pushover alert: {title}")
    try:
        pushover_client.send_message(message, title=title, priority=priority, sound='siren')
    except Exception as e:
        print(f"Failed to send Pushover alert: {e}")


def make_emergency_call():
    """Make emergency phone call using Twilio when driver doesn’t respond."""
    print("Making EMERGENCY CALL via Twilio...")
    try:
        twiml_url = "https://handler.twilio.com/twiml/EH97f61d52ddbe0a01a8d11ad4428b8f72"
        call = twilio_client.calls.create(
            url=twiml_url,
            to=EMERGENCY_CONTACT_NUMBER,
            from_=TWILIO_PHONE_NUMBER
        )
        print(f"Emergency call initiated with SID: {call.sid}")
    except Exception as e:
        print(f"Failed to make Twilio call: {e}")


# ------------------ GPIO Setup for Local Alerts ------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
SPEAKER_PIN = 21
LED_PIN = 17
GPIO.setup(SPEAKER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)
pwm = GPIO.PWM(SPEAKER_PIN, 440)


# ------------------ Helper Functions ------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)


# ------------------ Load Known Driver Faces ------------------
print("Loading known faces...")
known_face_encodings = []
known_face_names = []

for filename in os.listdir("known_faces"):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        try:
            image = face_recognition.load_image_file(f"known_faces/{filename}")
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])
        except IndexError:
            print(f"WARNING: No face found in {filename}. Skipping.")

print(f"Loaded {len(known_face_names)} authorized drivers: {known_face_names}")


# ------------------ Thresholds ------------------
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 10
YAWN_MAR_THRESH = 0.5
YAWN_CONSEC_FRAMES = 15
NO_FACE_CONSEC_FRAMES = 25


# ------------------ State Variables ------------------
eye_flag, yawn_flag, no_face_flag = 0, 0, 0
drowsiness_start_time = None
call_made_for_event = False


# ------------------ Initialize Dlib & Camera ------------------
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

current_driver = "Authenticating..."
guest_alert_sent = False

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
print("Camera started. Press 'q' to quit.")

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)


# ------------------ Main Loop ------------------
while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    alert_condition_met = False
    face_found = len(subjects) > 0

    # ---- Case 1: No face detected (Driver distracted) ----
    if not face_found:
        no_face_flag += 1
        if no_face_flag >= NO_FACE_CONSEC_FRAMES:
            alert_condition_met = True
            cv2.putText(frame, "DISTRACTION ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        no_face_flag = 0
        subject = subjects[0]

        # ---- Driver Authentication ----
        face_locations = [(subject.top(), subject.right(), subject.bottom(), subject.left())]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) > 0:
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            name = "Unknown"
            if True in matches:
                name = known_face_names[matches.index(True)]

            if name != current_driver:
                current_driver = name
                print(f"Driver identified as: {current_driver}")
                guest_alert_sent = False

        cv2.putText(frame, f"Driver: {current_driver}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ---- Alert if Unauthorized Driver ----
        if current_driver == "Unknown" and not guest_alert_sent:
            send_phone_alarm("An unauthorized driver is operating the vehicle.",
                             title="Security Alert", priority=0)
            guest_alert_sent = True

        # ---- Detect Drowsiness ----
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # ---- Eye Closure ----
        if ear < EYE_AR_THRESH:
            eye_flag += 1
            if eye_flag >= EYE_AR_CONSEC_FRAMES:
                alert_condition_met = True
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            eye_flag = 0

        # ---- Yawning ----
        if mar > YAWN_MAR_THRESH:
            yawn_flag += 1
            if yawn_flag >= YAWN_CONSEC_FRAMES:
                alert_condition_met = True
                cv2.putText(frame, "YAWN ALERT!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            yawn_flag = 0

    # ------------------ Handle Alerts ------------------
    if alert_condition_met:
        GPIO.output(LED_PIN, GPIO.HIGH)
        pwm.start(50)

        if drowsiness_start_time is None:
            print("Alert event started. Sending phone alarm.")
            drowsiness_start_time = time.time()
            send_phone_alarm("DROWSINESS ALERT! Please respond.")  # triggers cloud alert

        elapsed_time = time.time() - drowsiness_start_time
        if elapsed_time > 20 and not call_made_for_event:
            make_emergency_call()
            call_made_for_event = True

    else:
        GPIO.output(LED_PIN, GPIO.LOW)
        pwm.stop()
        if drowsiness_start_time is not None:
            print("Alert event ended.")
        drowsiness_start_time = None
        call_made_for_event = False

    # ------------------ Display ------------------
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
pwm.stop()
GPIO.cleanup()
