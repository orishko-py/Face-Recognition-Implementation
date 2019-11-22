import numpy as np 
import cv2
import os 

# ===== AUXILIARY FUNCTIONS =====

def listdir_nohidden(path):
    """
    had to make sure to ignore hidden files 
    https://stackoverflow.com/questions/7099290/how-to-ignore-hidden-files-using-os-listdir
    """
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def data_prep():
    # this function labels every face in each existing folder
    images = []
    names =[]
    names_dict = {}
    # collect all existing names from the faces folder
    all_data = [f for f in listdir_nohidden('faces/')]
    for i, name in enumerate(all_data):
        names_dict[i] = name
        for img in os.listdir('faces/'+ name):
            images.append(cv2.imread('faces/' + name + '/' + img, 0))
            names.append(i)
    return (images, np.array(names), names_dict)


def face_crop(image, faces):
    # crop the box of the image 
    cropped_face = []
    for (x,y,w,h) in faces:
        cropped_w = int(0.2 * w / 2)
        cropped_face.append(frame[y: y + h, x + cropped_w: x + w - cropped_w]) 
    return cropped_face

def face_normalization(images):
    # normalize intensity of the face for better recognition of features

    images_norm = []
    for img in images:
        images_norm.append(cv2.equalizeHist(img))
    return images_norm

def face_resize(images, size=(60,60)):
    # fix the size of the image 
    images_scaled = []
    for img in images:
        scaled = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        images_scaled.append(scaled)
    return images_scaled

def prepare_images(frame, faces):
    #combine the functions above
    cropped_faces = face_crop(frame, faces)
    resized_faces = face_resize(cropped_faces)
    final = face_normalization(resized_faces)
    return final

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

    def __init__(self):
        self.classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def locate(self, image, biggest_only=True):
        # variable definitions
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH
        coordinate_set = self.classifier.detectMultiScale(image,
                                                          scaleFactor = scale_factor,
                                                          minNeighbors=min_neighbors,
                                                          minSize=min_size,
                                                          flags = flags)
        return coordinate_set


# ===== Face Recognition Class =====
class Recognize(object):

    def __init__(self, images, names):
        # initialize and train
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(images, names)        

    def predict(self, normal_image):
        collector = cv2.face.StandardCollector_create()
        self.recognizer.predict_collect(normal_image, collector)
        confidence = collector.getMinDist()
        prediction = collector.getMinLabel()
        return confidence, prediction


# ===== Initialization of Variables =====
keepOn = True
video = VideoFeed()
face_detector = FaceDetector()

images, names, names_dict = data_prep()
recognizer = Recognize(images, names)

print('Success')

# ===== Display Loop =====
while(keepOn):
    frame = cv2.cvtColor(video.frame_getter(), cv2.COLOR_BGR2GRAY)
    faces = face_detector.locate(frame)

    # run for at least one face detected
    if len(faces):
        final = prepare_images(frame, faces)

        cv2.imshow('output', final[0])
        for j ,face in enumerate(final):

            # predicting the person
            confidence, prediction = recognizer.predict(face)

            # setting a custom threshold for our confidence level
            threshold = 130

            # add predicted name of the located face next to the box,
            # return unknown otherwise
            if confidence < threshold:
                cv2.putText(frame, 
                            names_dict[prediction] + ': {0:.2f}'.format(confidence), 
                            (faces[j][0], faces[j][1] - 5), 
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                            1, (255,255,255))
            else: 
                cv2.putText(frame, 'Unknown', 
                            (faces[j][0], faces[j][1] - 5), 
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                            1, (255,255,255))
    
    # add the box as before
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)

    cv2.imshow('output', frame)

    #exit case
    if cv2.waitKey(20) & 0xFF == ord('q'):
        video.__del__()
        keepOn=False

