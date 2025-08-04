import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Email Configuration
EMAIL_ADDRESS = "sairamdeshetty0115@gmail.com"  # Your email address
EMAIL_PASSWORD = "hqgu braa uhfu kvuk"  # Your email password or app password
TO_EMAIL_ADDRESS = "deshettysairam0115@gmail.com"  # Recipient's email address

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Argument parser for confidence and threshold values
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="Threshold for non-maxima suppression")
args = vars(ap.parse_args())

# Load YOLO configuration and weights
labels_path = "obj.names"
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"

LABELS = open(labels_path).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Open video file
video_path = "inp2.mp4"
cap = cv2.VideoCapture(video_path)

# Create output folder for detected frames
output_folder = "detected_frames"
os.makedirs(output_folder, exist_ok=True)

# Variables for frame dimensions and movement tracking
(W, H) = (None, None)
prev_landmarks = None
fighting_threshold = 0.2
falling_threshold = 0.5
frame_count = 0
frame_skip = 2  # Process every 2nd frame for speed

# Timestamp for the last alert
last_alert_time = 0
alert_interval = 60  # Send email every 60 seconds at most


def calculate_movement_speed(current_landmarks, prev_landmarks):
    """Calculate the average movement speed of landmarks."""
    if prev_landmarks is None:
        return 0
    return np.mean([
        np.linalg.norm(np.array([c.x, c.y, c.z]) - np.array([p.x, p.y, p.z]))
        for c, p in zip(current_landmarks, prev_landmarks)
    ])


def detect_fighting_or_falling(current_landmarks, prev_landmarks):
    """Detect abnormal activities: fighting or falling."""
    global frame_count
    if prev_landmarks is None:
        return "Normal Activity"

    movement_speed = calculate_movement_speed(current_landmarks, prev_landmarks)
    pelvis_y_diff = abs(current_landmarks[24].y - prev_landmarks[24].y)
    shoulder_y_diff = abs(current_landmarks[11].y - prev_landmarks[11].y)

    if pelvis_y_diff > falling_threshold or shoulder_y_diff > falling_threshold:
        frame_count += 1
        if frame_count > 5:
            return "Abnormal Activity Detected"
    else:
        frame_count = 0

    hand_speed = max(
        calculate_movement_speed([current_landmarks[15]], [prev_landmarks[15]]),
        calculate_movement_speed([current_landmarks[16]], [prev_landmarks[16]])
    )
    if movement_speed > fighting_threshold and hand_speed > fighting_threshold:
        return "Abnormal Activity Detected"

    return "Normal Activity"


def send_email(subject, body, image_path):
    """Send an email with the activity and attached image."""
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = TO_EMAIL_ADDRESS
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        with open(image_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
        msg.attach(part)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, TO_EMAIL_ADDRESS, msg.as_string())
        print("[INFO] Email sent successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")


frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % frame_skip != 0:
        continue

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    abnormal_detected = False
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{LABELS[class_ids[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if LABELS[class_ids[i]] in ["gun", "fire"]:
                abnormal_detected = True

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        current_landmarks = results.pose_landmarks.landmark
        activity = detect_fighting_or_falling(current_landmarks, prev_landmarks)
        prev_landmarks = current_landmarks
        color = (0, 255, 0) if activity == "Normal Activity" else (0, 0, 255)
        cv2.putText(frame, activity, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if activity != "Normal Activity":
            abnormal_detected = True

    if abnormal_detected:
        frame_path = os.path.join(output_folder, f"frame_{frame_counter}.jpg")
        cv2.imwrite(frame_path, frame)

        # Send email alert if the alert interval has passed
        current_time = time.time()
        if current_time - last_alert_time > alert_interval:
            send_email(
                subject="Abnormal Activity Detected",
                body=f"An abnormal activity has been detected: {activity}. Please find the image attached.",
                image_path=frame_path
            )
            last_alert_time = current_time

    cv2.imshow('Intelligent video surveillance using deep learning', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
