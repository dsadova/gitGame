import tarfile
import cv2
import dlib
import numpy as np

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

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
    y0 = frameHeight
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            if y1<y0:
                y0 = y1
                del bboxes[:]
                bboxes.append([x1, y1, x2, y2])
    cv2.rectangle(frameOpencvDnn, (bboxes[0][0], bboxes[0][1]),
                  (bboxes[0][2], bboxes[0][3]), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

#f = open("names.txt", "r")

#line = f.readline()
#i = 1
#while line:
    #str1 = "originalPics/" + line.strip('\n') + ".jpg"
str1 = "test" + ".jpg"
image = cv2.imread(str1)
height1, width1, channels = image.shape
image1 = image_resize(image, width = 1024)
cv2.imwrite("test1"+".jpg", image1)
height2, width2, channels = image1.shape
k = width1/width2
dnnFaceDetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
faceRects = dnnFaceDetector(image1, 0)
cv_img, cvboxes = detectFaceOpenCVDnn(net, image1)
cv2.imwrite("test2"+".jpg", cv_img)
crop_img = cv_img[cvboxes[0][1]:cvboxes[0][3], cvboxes[0][0]:cvboxes[0][2]]
h = k*(cvboxes[0][2]-cvboxes[0][0])
w = k* (cvboxes[0][3]-cvboxes[0][1])
cv2.imwrite("cropped.jpg", crop_img)
crop_img1 = image[k*cvboxes[0][1] - h/2:height1, k*cvboxes[0][0]-w:k*cvboxes[0][2]+w]
#    str2 = "cropped/" + line.strip('\n') + ".jpg"
cv2.imwrite("cropped/" + "test" + ".jpg", crop_img1)
#    i+=1
#    line = f.readline()
#f.close()