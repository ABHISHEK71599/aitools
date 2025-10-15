#!/usr/bin/env python3
"""
video_face_detector.py
-------------------------------------
A simple free face detector for videos.

Usage:
    python3 video_face_detector.py input.mp4

Requirements:
    pip install opencv-python
    (FFmpeg not required — OpenCV reads video frames directly)

Output:
    - Saves face images to ./faces/
    - Prints 'person found' if any faces detected, otherwise 'no person found'
"""

import cv2
import os
import sys
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 video_face_detector.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    # Prepare output folder
    faces_dir = os.path.join(os.path.dirname(video_path), "faces")
    os.makedirs(faces_dir, exist_ok=True)

    # Load face detector (Haar cascade)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: could not load Haar cascade")
        sys.exit(1)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        sys.exit(1)

    frame_count = 0
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        frame_count += 1
        # Process every nth frame to save time
        if frame_count % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            pad = int(0.2 * w)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_img = frame[y1:y2, x1:x2]
            out_path = os.path.join(faces_dir, f"face_{face_count:04d}.jpg")
            cv2.imwrite(out_path, face_img)
            face_count += 1

    cap.release()

    if face_count > 0:
        print("person found")
    else:
        print("no person found")

    # Optional: print result summary as JSON
    result = {
        "video": video_path,
        "faces_found": face_count,
        "faces_dir": faces_dir
    }
    with open(os.path.join(faces_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
