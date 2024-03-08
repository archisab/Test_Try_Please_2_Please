import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

from flask import Flask, jsonify, request


detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0
recognized_letter = ""

# labels = ["A", "B", "C"]

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "U", "V", "W", "X", "Y"]

prev_letters = []


def load_prev_letters():
    
    try:
        
        with open("prev_letters_tracker.txt", "r") as file:
            
            return [line.strip() for line in file.readlines()]
            
    except FileNotFoundError:
        
        return []


def save_prev_letters(prev_letters):
    
    with open("prev_letters_tracker.txt", "w") as file:
        
        for letter in prev_letters:
            
            file.write(f"{letter}\n")


def image_process(img):    

    global recognized_letter

    global prev_letters

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:

        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)


        if len(prev_letters) == 1: # 2

            current_letter = labels[index]
            
            if all(letter == current_letter for letter in prev_letters):

                recognized_letter = current_letter
            
            prev_letters.clear()
            
        else:
            
            prev_letters.append(labels[index])


    return recognized_letter


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def sending_the_image():

    global prev_letters

    global recognized_letter


    if 'file' not in request.files:

        return jsonify(error="Please Try Again. The Image does not currently exist here.")
    

    file = request.files.get('file')

    img_path = "test_image.png"

    file.save(img_path)

    image = cv2.imread(img_path)


    prev_letters = load_prev_letters()

    result = image_process(image)

    if result != "":

        recognized_letter = ""

    save_prev_letters(prev_letters)


    return jsonify(prediction=result)


if __name__ == '__main__':

    app.run(debug=False, host='0.0.0.0')
