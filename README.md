# Face-Recognition-Implementation
The code for creating a data set of images containing a person's face and then using it to recognize any person in the data set on camera. The implementation uses openCV and there is a version that uses mtcnn (https://github.com/ipazc/mtcnn), but it was quite slow in a live setting.
In order to collect images of person's name, run facecapture.py locally and it will create a folder with the name you input if a folder 'faces' is located in the same directory. 
Once the data set is collected, run main_version.py or mtcnn_version.py from the same directory. 
