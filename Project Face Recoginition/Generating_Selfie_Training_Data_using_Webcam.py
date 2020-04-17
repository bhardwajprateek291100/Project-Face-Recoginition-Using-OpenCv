# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2

import numpy as np

## Initialise camera
cap = cv2.VideoCapture(0)

## Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


skip = 0

face_data = []             ## store the face data captured

dataset_path = './data/'    ## path of file to store the detected face data

face_section = 0    ## just for globalise

## input of name of person
file_name = input("Enter the name of the person : ")

while True:

	ret, frame = cap.read()

	if ret == False:
		continue

	# gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)   ## list of tuples contain x,y,w,h for each face

	#print(faces)

	## sorting faces on the area of frame in descending order consideringly the largest face
	faces = sorted(faces, key = lambda f:f[2]*f[3])


	## pick the last face as it is the largest face according to area f rectangle w*h
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)    ## frame ,boundary of rectangele, colour, thickness of frame

		## Extract i.e Crop out the required face : Region Of Interest
		offset = 10     ## adding 10 pixels in all directions for padding

		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]    ## frame has y as first attribute and x as second one

		face_section = cv2.resize(face_section,(100,100))        ## resizing the face captured

		## appending the captured face data section
		skip += 1
		if skip%10 == 0:
			face_data.append(face_section)
			print(len(face_data))


	cv2.imshow("Frame", frame)
	cv2.imshow("Face Section", face_section)


	## exiting
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

	if skip >= 200:
		break


## Convert our face list array into a numpy array
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

## save this file into system
np.save(dataset_path + file_name + '.npy', face_data)

print("Data Successfully save")


cap.release()
cv2.destroyAllWindows()