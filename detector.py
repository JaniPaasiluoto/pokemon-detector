import time

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
from imutils.video import VideoStream

vs = VideoStream(src=0).start()
time.sleep(2.0)

print("[INFO] taking pic in 3...")
time.sleep(3.0)
frame = vs.read()
image = imutils.resize(frame, width=260)
output = image.copy()
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model("assets/pokedex.model")
print("[INFO] loading labels...")
lb = pickle.loads(open("assets/lb.pickle", "rb").read())


# classify the input image
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

# build the label and draw the label on the image
label = "{}: {:.2f}%".format(label, proba[idx] * 100)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

cv2.imwrite("results/" + label + ".jpg", output)

# show the output image
print("[INFO] {}".format(label))
# cv2.imshow("Output", output)
# cv2.waitKey(0)
