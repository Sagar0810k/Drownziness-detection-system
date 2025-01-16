import cv2
import dlib
from scipy.spatial import distance
import winsound
import time

# Define EAR calculation function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"H:\PROJECTS OF PYTHON\Drownziness system\shape_predictor_68_face_landmarks.dat")


# Indices for eye landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
EAR_THRESHOLD = 0.3
CHECK_INTERVAL = 3  # Check every 3 seconds

# Initialize variables
cap = cv2.VideoCapture(0)
drowsy_count = 0
last_check_time = time.time()
alarm_triggered = False
ear_history = []

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = [landmarks.part(i) for i in LEFT_EYE]
        right_eye = [landmarks.part(i) for i in RIGHT_EYE]

        # Convert landmarks to lists of (x, y) tuples
        left_eye = [(p.x, p.y) for p in left_eye]
        right_eye = [(p.x, p.y) for p in right_eye]

        # Calculate EAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2
        ear_history.append(avg_ear)

        # Smooth the EAR signal using the last 10 frames
        if len(ear_history) > 10:
            ear_history.pop(0)
        smooth_ear = sum(ear_history) / len(ear_history)

        # Check drowsiness every 3 seconds
        current_time = time.time()
        if current_time - last_check_time > CHECK_INTERVAL:
            if smooth_ear < EAR_THRESHOLD:
                drowsy_count += 1
                alarm_triggered = True
                # Beep sound: 1000Hz frequency, 500ms duration
                winsound.Beep(1000, 500)
            else:
                alarm_triggered = False
            last_check_time = current_time

        # Draw bounding box and text
        for point in left_eye:
            cv2.circle(frame, point, 1, (0, 255, 0), -1)
        for point in right_eye:
            cv2.circle(frame, point, 1, (0, 255, 0), -1)

        box_color = (0, 0, 255) if alarm_triggered else (0, 255, 0)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), box_color, 2)
        cv2.putText(frame, f"Drowsy Count: {drowsy_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
