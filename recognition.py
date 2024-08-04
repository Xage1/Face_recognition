import cv2
import os
import numpy as np
from PIL import Image

# Paths for the training data
face_cascade_path = 'D:/DEV OPS/face/haarcascade_frontalface_default.xml'
training_data_path = 'D:/DEV OPS/face/training_data.yml'
images_path = 'D:/DEV OPS/face/images/'

# Ensure the images directory exists
if not os.path.exists(images_path):
    os.makedirs(images_path)

# Download the Haarcascade file if missing
def download_haarcascade(url, save_path):
    import urllib.request
    urllib.request.urlretrieve(url, save_path)
    print(f"Downloaded Haarcascade file to: {save_path}")

# Check and download the Haarcascade file if it doesn't exist
if not os.path.isfile(face_cascade_path):
    print(f"Haarcascade file not found: {face_cascade_path}")
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    download_haarcascade(haarcascade_url, face_cascade_path)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Ensure the face cascade is loaded correctly
if face_cascade.empty():
    raise Exception(f"Failed to load Haarcascade from {face_cascade_path}. Ensure it contains valid data.")

# Function to capture face images for training
def capture_images(label, count=20):
    cap = cv2.VideoCapture(0)
    captured_count = 0
    print(f"Capturing images for label {label}. Press 'q' to stop.")
    
    while captured_count < count:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            captured_count += 1
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(images_path, f"{label}.{captured_count}.jpg")
            cv2.imwrite(img_path, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Capturing Images', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {captured_count} images for label {label}.")

# Function to train the recognizer and save the model
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    face_ids = []

    for root, dirs, files in os.walk(images_path):
        for file in files:
            if file.endswith("jpg"):
                img_path = os.path.join(root, file)
                gray_img = Image.open(img_path).convert('L')
                img_numpy = np.array(gray_img, 'uint8')
                face_id = int(os.path.split(img_path)[-1].split('.')[0])
                faces = face_cascade.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    face_ids.append(face_id)

    recognizer.train(face_samples, np.array(face_ids))
    recognizer.save(training_data_path)
    print(f"Training completed and saved to {training_data_path}")

# Main script execution
if __name__ == "__main__":
    # Capture images for a few labels (e.g., 1, 2, 3)
    capture_images(label=1)
    capture_images(label=2)
    capture_images(label=3)

    # Train the recognizer
    train_recognizer()

    # Load the pre-trained face recognition model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(training_data_path)

    # Initialize the video capture for recognition
    video_capture = cv2.VideoCapture(0)

    while True:
        # Read the current frame from the video capture
        ret, frame = video_capture.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Recognize the face
            try:
                label, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                # Display the name of the recognized person
                if confidence < 100:
                    name = "Person " + str(label)
                else:
                    name = "Unknown"
            except cv2.error as e:
                name = "Unknown"
                print(f"Error during prediction: {e}")

            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Recognizer', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()