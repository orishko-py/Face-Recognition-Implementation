"""
The following file is used to create a dataset of a face using webcam
it saves pictures in a folder called faces, that is located in the 
same directory as this file.
"""

import cv2
import numpy as np 
import os

# crop the box of the image 
def face_crop(image, faces):
    cropped_face = []
    for (x,y,w,h) in faces:
        cropped_w = int(0.2 * w / 2)
        cropped_face.append(frame[y: y + h, x + cropped_w: x + w - cropped_w]) 
    return cropped_face

# normalize intensity of the face for better recognition of features
def face_normalization(images):
    images_norm = []
    for img in images:
        images_norm.append(cv2.equalizeHist(img))
    return images_norm

def face_resize(images, size=(60,60)):
    images_scaled = []
    for img in images:
        scaled = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        images_scaled.append(scaled)
    return images_scaled

# ===== Video Camera Class =====
class VideoFeed(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        print(self.video.isOpened())

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()
    
    def frame_getter(self):
        _, frame = self.video.read()
        return frame

# ===== Face Detector Class =====
class FaceDetector(object):
    def __init__(self, xml):
        self.classifier = cv2.CascadeClassifier(xml)

    def locate(self, image, biggest_only=True):
        # variable definitions
        scale_factor = 1.3
        min_neighbors = 5
        min_size = (30, 30)
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH
        coordinate_set = self.classifier.detectMultiScale(image,
                                                          scaleFactor = scale_factor,
                                                          minNeighbors=min_neighbors,
                                                          minSize=min_size,
                                                          flags = flags)
        return coordinate_set

# Initialization 
keepOn = True
video = VideoFeed()
face_detector = FaceDetector('haarcascade_frontalface_default.xml')

folder = 'faces/' + input('Name:')
cv2.namedWindow('Data Set Building', cv2.WINDOW_AUTOSIZE)

#the following obtains 20 / a number of pictures of a persons faces
if not os.path.exists(folder):
    os.mkdir(folder)
    count = 0
    timer = 0

    while count < 20:
        frame = cv2.cvtColor(video.frame_getter(), cv2.COLOR_BGR2GRAY)
        faces = face_detector.locate(frame)
        if len(faces) and timer % 700 == 50:
            cropped_faces = face_crop(frame, faces)
            resized_faces = face_resize(cropped_faces)
            final = face_normalization(resized_faces)

            # save the file in the appropriate directory
            cv2.imwrite(folder + '/' + str(count)+ '.jpg', final[0])

            # notify how many snapshots were taken so far
            cv2.imshow('Pictures collected: ' +str(count), final[0])
            count += 1
        cv2.imshow('output', frame)
        cv2.waitKey(50)
        timer +=50
    cv2.destroyAllWindows()
