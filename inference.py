import mediapipe as mp
import cv2
import os
from csv import writer
import time
from helpers import scale_single_coord, predict
from nets import HandNet, HandNet2
from dataAugmenter import augment_one

import pygame
pygame.mixer.init()

import threading
sound_playing = threading.Event()
isSTOP = False
global_thread_mutex = threading.Lock()
song = None
CONFIDENCE = 3
def sound():
    global sound_playing
    music_dict = {
        "thumbs up": "./public/thumbs_up.mp3",
        "peace sign": "./public/peace_sign.mp3",
        "euro footballer": "./public/euro_footballer.mp3",
        "kpop heart": "./public/kpop_heart.mp3",
        "what the sigma": "./public/what_the_sigma.mp3",
    }
    while not isSTOP:
        # sound_playing.wait()  # Wait until the event is set
        if sound_playing.is_set():
            print("Playing sound...")
            
            pygame.mixer.music.load(music_dict[song])
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if not sound_playing.is_set():
                    break  # Stop playing if the flag is cleared
                time.sleep(0.1)
            if not sound_playing.is_set():
                continue
            pygame.mixer.music.stop()  # Ensure the sound is stopped
            sound_playing.clear()
        time.sleep(0.1)
            


t1 = threading.Thread(target=sound)
t1.start()

model = HandNet2()
model.setup("./good_models/hand2_model_0_scaled_epoch12_lr0.001_bs_32.pth")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam successfully opened.")
counter = 0
same_as_last = 0
old_res = None
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
            row = []
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])
            row = augment_one(row)
            res = predict(model, row)
            if res:
                print(res)
                counter = 0
                # Detected a new output, do things
                if old_res != res:
                    old_res = res
                    same_as_last = 0
                elif same_as_last < CONFIDENCE and old_res == res:
                    same_as_last += 1
                    if same_as_last == CONFIDENCE:
                        print("NEW OUTPUT!")
                        with global_thread_mutex:
                            song = res
                            sound_playing.clear()
                        pygame.mixer.music.stop()
                        time.sleep(0.2)
                        with global_thread_mutex:
                            sound_playing.set()
            else: # Reset the timer
                counter += 1
            time.sleep(0.05)
        else:
            counter += 1
        
        if counter > 5:
            old_res = None
            
            

        cv2.imshow('Hand Tracking', image)

        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            isSTOP = True
            break
            
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
t1.join()