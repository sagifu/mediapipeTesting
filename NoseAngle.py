
# Import Libraries
import cv2
import time
import mediapipe as mp
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'.    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

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

# Initializing landmark index count
Initial_angle_flag = True

# Initializing image size
width,height = 800,600

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
	frame = cv2.resize(frame, (width,height))

	# Converting the from BGR to RGB
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Making predictions using holistic model
	# To improve performance, optionally mark the image as not writeable to
	# pass by reference.
	image.flags.writeable = False
	results = holistic_model.process(image)
	image.flags.writeable = True

	# Extract pose landmarks - normalized X and Y of screen width and height, Z the distance from screen, and visibility.
	pose_landmark = results.pose_landmarks

	# Save initial values of facial landmarks
	# Previous code verified:
	# Landmark 1 - Nose
	# Landmark 2 - Inner left eye
	# Landmark 3 - Middle left eye
	# Landmark 4 - Outer left eye
	# Landmark 5 - Inner right eye
	# Landmark 6 - Middle right eye
	# Landmark 7 - Outer right eye
	# Landmark 8 - Left cheek bone
	# Landmark 9 - Right cheek bone
	n_landmarks = 9
	if Initial_angle_flag:
		initial_pose = pose_landmark.landmark[:n_landmarks]
		# initial_angle = 
		Initial_angle_flag = False

	##########

	# # Move all landmarks to align with nose
	# align_vec = [initial_pose.landmark[0].x - pose_landmark.landmark[0].x, 
	# 			 initial_pose.landmark[0].y - pose_landmark.landmark[0].y, 
	# 			 initial_pose.landmark[0].z - pose_landmark.landmark[0].z]
	# for i,x in enumerate(pose_landmark.landmark):
	# 	pose_landmark.landmark[i].x -= align_vec[0]
	# 	pose_landmark.landmark[i].y -= align_vec[1]
	# 	pose_landmark.landmark[i].z -= align_vec[2]

	# print(pose_landmark.landmark)
	# print(' ')
	# print(initial_pose)
	# capture.release()
	# cv2.destroyAllWindows()
	# break

	##########

	# Converting back the RGB image to BGR
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	
	# Drawing Pose Land Marks
	mp_drawing.draw_landmarks(
	image,
	results.pose_landmarks
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
