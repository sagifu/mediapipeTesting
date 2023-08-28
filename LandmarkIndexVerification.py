
# Import Libraries
import cv2
import time
import mediapipe as mp
import numpy as np

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Initializing current time and precious time for calculating the FPS
previousTime = time.time()
currentTime = 0

# Initializing lendmark index count
landmark_idx = 0


while True:
	# capture frame by frame
	ret, frame = capture.read()

	# print(type(frame))
	if frame is None:
		capture.release()
		cv2.destroyAllWindows()
		break

	# flip image to mirror
	frame = cv2.flip(frame, 1)

	# resizing the frame for better view
	frame = cv2.resize(frame, (800, 600))

	# Converting the from BGR to RGB
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Making predictions using holistic model
	# To improve performance, optionally mark the image as not writeable to
	# pass by reference.
	image.flags.writeable = False
	results = holistic_model.process(image)
	image.flags.writeable = True

	pose_landmark = results.pose_landmarks
	for i,x in enumerate(pose_landmark.landmark):
		if i!=landmark_idx:
			pose_landmark.landmark[i].visibility = 0

	##########

	# print(results.pose_landmarks.landmark)
	# print(dir(results.pose_landmarks.landmark))
	# capture.release()
	# cv2.destroyAllWindows()
	# break

	##########


	# Converting back the RGB image to BGR
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	
	# Drawing Pose Land Marks
	mp_drawing.draw_landmarks(
	image,
	pose_landmark
	)

	# Calculating the FPS
	currentTime = time.time()
	if (currentTime-previousTime) > 5:
		landmark_idx += 1
		previousTime = currentTime
	
	# Displaying FPS on the image
	cv2.putText(image, str(landmark_idx+1)+" Landmark", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

	# Display the resulting image
	cv2.imshow("Facial Landmarks Order", image)

	# Termination
	k = cv2.waitKey(1)
	if k == 27: # esc
		capture.release()
		cv2.destroyAllWindows()
		break
