import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

plot_mesh_flag = False

facmesh = mp.solutions.face_mesh
face = facmesh.FaceMesh(static_image_mode=True,
	min_tracking_confidence=0.6,
	min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

while True:
	_, frm = cap.read()
	frm = cv2.flip(frm, 1)
	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	if plot_mesh_flag:
		op = face.process(rgb)
		if op.multi_face_landmarks:
			for i in op.multi_face_landmarks:
				draw.draw_landmarks(
					frm, 
					i, 
					facmesh.FACEMESH_CONTOURS, 
					landmark_drawing_spec=draw.DrawingSpec(circle_radius=1)
				)

	cv2.imshow("window", frm)

	k = cv2.waitKey(1)
	if k == 32: # spacebar
		plot_mesh_flag = not(plot_mesh_flag)
	elif k == 27: # esc
		cap.release()
		cv2.destroyAllWindows()
		break


		