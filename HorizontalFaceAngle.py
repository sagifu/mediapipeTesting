
# Import Libraries
import cv2
import time
import mediapipe as mp
import numpy as np
 

# Initial values of facial landmarks
# Previous code verified:
# Landmark 1 - Nose
# Landmark 2 - Left cheek bone
# Landmark 5 - Right cheek bone
n_landmarks = [1,8,9]

def calculate_plane_normal(points):
    # Given three points on a plane, calculate the normal vector of the plane
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def calculate_angle_between_planes(points1):
    # Calculate the angle between two planes defined by their normal vectors
    normal1 = calculate_plane_normal(points1)
    normal2 = np.array([0, 1, 0])
    
    dot_product = np.dot(normal1, normal2)
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

# Function to extract landmarks of interest and align by first landmark
def extract_and_align(pose_input, sc_width, sc_height):
	new_pose = list()
	for i in n_landmarks:
		new_pose.append([
			pose_input.landmark[i].x * sc_width, 
			pose_input.landmark[i].y * sc_height,
			pose_input.landmark[i].z
			])
	new_pose = np.array(new_pose)
	align_vec = np.array([0 - new_pose[0][0], 
		 0 - new_pose[0][1], 
		 0 - new_pose[0][2]])
	for i,x in enumerate(new_pose):
		new_pose[i] += align_vec
	return new_pose


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

# # Initializing landmark index count
# initial_angle_flag = True

# Initializing image size
width,height = 800,600

while True:
	# capture frame by frame
	ret, frame = capture.read()

	# Terminate loop if recorded video has ended
	if frame is None:
		break

	# Flip image to mirror
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

	# Extract current pose
	current_pose = extract_and_align(results.pose_landmarks, width, height)

# 	##########

# 	# print(pose_landmark.landmark)
# 	# print(' ')
# 	# print(initial_pose)
# 	# print(' ')
# 	# print(current_pose)
# 	# print(' ')
# 	# print(distance(initial_pose, current_pose))
# 	# capture.release()
# 	# cv2.destroyAllWindows()
# 	# break

# 	##########

	facial_degree = calculate_angle_between_planes(current_pose)

	# Converting back the RGB image to BGR
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	
	# Drawing Pose Land Marks
	mp_drawing.draw_landmarks(
	image,
	results.pose_landmarks
	)
	
	# Displaying angle on the image
	cv2.putText(image, f"Deg: {round(facial_degree, 2)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

	# Display the resulting image
	cv2.imshow("Facial Landmarks Order", image)

	# Termination
	k = cv2.waitKey(1)
	if k == 27: # esc
		break

capture.release()
cv2.destroyAllWindows()