import cv2
import time
from cvzone.ClassificationModule import Classifier
import json

folder = "Data/Saved"
counter = 0
selections = []
selected = False
classifier = Classifier("converted_keras/keras_model.h5", "converted_keras/labels.txt")
labels = ["empty", "occupied"]
check_enabled = False
last_check_time = time.time()

def save_selections():
    with open('selections.json', 'w') as f:
        json.dump(selections, f)

def load_selections():
    global selections
    try:
        with open('selections.json', 'r') as f:
            selections = json.load(f)
    except FileNotFoundError:
        selections = []
def mouse_event(event, x, y, flags, param):
    global selections, selected

    if event == cv2.EVENT_LBUTTONDOWN:
        selected = True
        selections.append({'start_x': x, 'start_y': y, 'end_x': x, 'end_y': y, 'occupied': True})

    elif event == cv2.EVENT_MOUSEMOVE and selected:
        if selections:
            selections[-1]['end_x'], selections[-1]['end_y'] = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        selected = False

# Load selections from file
load_selections()


# Open webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow('Webcam')
cv2.setMouseCallback('Webcam', mouse_event)

while True:
    ret, frame = cap.read()

    # Iterate through selections to draw rectangles and display cropped regions
    for sel in selections:
        if sel['occupied']:
            cv2.rectangle(frame, (sel['start_x'], sel['start_y']), (sel['end_x'], sel['end_y']), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (sel['start_x'], sel['start_y']), (sel['end_x'], sel['end_y']), (0, 255, 0), 2)
        cropped_region = frame[sel['start_y']:sel['end_y'], sel['start_x']:sel['end_x']]
        if cropped_region.shape[0] > 0 and cropped_region.shape[1] > 0:
            cv2.imshow(f'Cropped Region {selections.index(sel) + 1}', cropped_region)

    # Perform the check every 2 seconds if enabled
    if check_enabled and time.time() - last_check_time >= 2:
        for sel in selections:
            cropped_region = frame[sel['start_y']:sel['end_y'], sel['start_x']:sel['end_x']]
            if cropped_region.shape[0] > 0 and cropped_region.shape[1] > 0:
                prediction, index = classifier.getPrediction(cropped_region)
                if labels[index] == "empty":
                    sel['occupied'] = False
                elif labels[index] == "occupied":
                    sel['occupied'] = True
        last_check_time = time.time()  # Update the last check time

    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        check_enabled = not check_enabled  # Toggle the check state
        last_check_time = time.time()  # Reset the last check time
        for sel in selections:
            cropped_region = frame[sel['start_y']:sel['end_y'], sel['start_x']:sel['end_x']]
            if cropped_region.shape[0] > 0 and cropped_region.shape[1] > 0:
                cv2.imwrite(f'{folder}/Image_{time.time() + counter}.jpg', cropped_region)
                counter += 1
        save_selections()  # Save selections to file
        print("Cropped regions saved.")

    if key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
