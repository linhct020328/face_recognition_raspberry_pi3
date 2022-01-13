#run:  python pi_image.py --image images/069.jpg --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])


image = cv2.imread(args["image"])
fps = FPS().start()

image = imutils.resize(image, width=500)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
	minNeighbors=5, minSize=(30, 30))
boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
encodings = face_recognition.face_encodings(rgb, boxes)
names = []

for encoding in encodings:
	matches = face_recognition.compare_faces(data["encodings"],
		encoding)
	name = "Unknown"

	if True in matches:
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		name = max(counts, key=counts.get)

	names.append(name)

for ((top, right, bottom, left), name) in zip(boxes, names):
	cv2.rectangle(image, (left, top), (right, bottom),(0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.imshow("Image", image)
cv2.waitKey(0)

