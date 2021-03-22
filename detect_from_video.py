import time

from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2

print("[INFO] loading network...")
model = load_model("assets/pokedex.model")
print("[INFO] loading labels...")
lb = pickle.loads(open("assets/lb.pickle", "rb").read())

def detect_pokemon():
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        # grab the frame from the threaded video stream and resize
        # it to have a maximum width of 260 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=260)

        predLabel = classify(preprocess(frame))
        print(predLabel)

        cv2.imshow("Pokemon", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

def classify(image):
    # classify the input image
    proba = model.predict(image)[0]
    # return the class label with the largest predicted probability
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    return label

def preprocess(image):
	# preprocess the image
	image = cv2.resize(image, (96, 96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# return the pre-processed image
	return image

if __name__ == '__main__':
    detect_pokemon()