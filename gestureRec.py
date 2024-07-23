import mediapipe as mp
import cv2
import os
from csv import writer

# Create a directory if it doesn't exist to store the CSV file
if not os.path.exists('data'):
    os.makedirs('data')

csv_file = './data/data.csv'

# Check if the CSV file exists, create it if not
if not os.path.isfile(csv_file):
    with open(csv_file, 'w', newline='') as f:
        csv_writer = writer(f)
        # Write the header
        header = []
        for i in range(21):
            header.extend([f'Hand_Landmark_{i}_x', f'Hand_Landmark_{i}_y', f'Hand_Landmark_{i}_z'])
        header.append('label')
        csv_writer.writerow(header)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam successfully opened.")
counter = 0
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        # To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False
        results = hands.process(image)
        
        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            # Select the hand to record data from, prioritizing the right hand
            hand_landmarks = results.multi_hand_landmarks[0]
            if len(results.multi_handedness) > 1:
                for idx, hand_handedness in enumerate(results.multi_handedness):
                    if hand_handedness.classification[0].label == 'Right':
                        hand_landmarks = results.multi_hand_landmarks[idx]
                        break

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                # Draw index number and coordinates
                coord_text = f'ID:{idx} x:{landmark.x:.2f} y:{landmark.y:.2f} z:{landmark.z:.2f}'
                cv2.putText(image, coord_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

            # Capture the hand landmark data for the selected hand
            with open(csv_file, 'a', newline='') as f_object:
                writer_object = writer(f_object)
                
                row = []
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])

                # LABEL
                row.append(4)  # Add label column with value 1

                writer_object.writerow(row)
            counter += 1

        cv2.imshow('Hand Tracking', image)

        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            break
        elif counter >= 3000:
            break
            
cap.release()
cv2.destroyAllWindows()