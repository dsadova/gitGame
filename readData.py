import tarfile
import cv2
import dlib
import numpy as np
'''tar = tarfile.open("originalPics.tar.gz", "r:gz")
i = 0
for member in tar.getmembers():
     f = tar.extractfile(member)
     if f is not None:
         content = f.read()
         i+=1
print(i)'''
modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
conf_threshold = 0.7
def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

f = open("FDDB-fold-05.txt", "r")

line = f.readline()
i = 1
while line:
    str1 = "originalPics/" + line.strip('\n') + ".jpg"
    image = cv2.imread(str1)
    height, width, channels = image.shape

    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
    faceRects = dnnFaceDetector(image, 0)
    bboxes = []
    cv_img, cvboxes = detectFaceOpenCVDnn(net, image)
    '''for faceRect in faceRects:
        x1 = faceRect.rect.left()
        y1 = faceRect.rect.top()
        x2 = faceRect.rect.right()
        y2 = faceRect.rect.bottom()
        bboxes.append([x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), int(round(height / 150)), 8)
        cv2.imwrite("./recognizedPic_dlib/"+"a_"+ str(i) +".jpg", image)'''

    cv2.imwrite("./recognizedPic_cv/"+"a_"+ str(i) +".jpg", cv_img)
    i+=1
    line = f.readline()
f.close()

