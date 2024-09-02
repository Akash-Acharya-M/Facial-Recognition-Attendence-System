import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, time

# Directory where known faces are stored
KNOWN_FACES_DIR = 'known_faces'
# Directory to save unrecognized faces
UNRECOGNIZED_FACES_DIR = 'unrecognized_faces'
# File to store attendance
ATTENDANCE_FILE = 'attendance.csv'

# Ensure the unrecognized faces directory exists
if not os.path.exists(UNRECOGNIZED_FACES_DIR):
    os.makedirs(UNRECOGNIZED_FACES_DIR)

# Load known faces
known_face_encodings = []
known_face_names = []

# Function to load known faces
def load_known_faces():
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, filename)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
                    else:
                        print(f"No face found in {image_path}")

# Set the time window for marking attendance
ATTENDANCE_START_TIME = time(13, 46, 0)  # 11:51 AM
ATTENDANCE_END_TIME = time(13, 48, 10)   # 11:52 AM

# Function to mark attendance
def mark_attendance(name):
    with open(ATTENDANCE_FILE, 'a') as f:
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')
        f.write(f'{name},{date_string},{time_string}\n')

def save_unrecognized_face(image, timestamp):
    filename = f"unrecognized_{timestamp}.jpg"
    file_path = os.path.join(UNRECOGNIZED_FACES_DIR, filename)
    cv2.imwrite(file_path, image)

def main():
    # Load known faces
    print("Loading existing data...")
    load_known_faces()
    print(f"Loaded {len(known_face_names)} known faces.")

    # Initialize webcam
    print("Initializing webcam...")
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open video capture device.")
        return

    # Get and print camera properties
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"Camera initialized with resolution: {width}x{height}, FPS: {fps}")

    marked_names = set()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name not in marked_names:
                    now = datetime.now().time()
                    if ATTENDANCE_START_TIME <= now <= ATTENDANCE_END_TIME:
                        mark_attendance(name)
                        marked_names.add(name)
                        print(f"Marked attendance for {name}")
            else:
                # Save the unrecognized face
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unrecognized_face_image = frame[top:bottom, left:right]
                save_unrecognized_face(unrecognized_face_image, timestamp)
                print(f"Saved unrecognized face at {timestamp}")

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
