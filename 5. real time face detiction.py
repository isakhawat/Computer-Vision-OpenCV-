import numpy as np
import cv2

# We point OpenCV's CascadeClassifier function to where our
# classifier (XML file format) is stored
face_classifier = cv2.CascadeClassifier('C:\\Users\\Sabakat\\Desktop\\Haarcascades\\haarcascade_frontalface_default.xml')

# Load our image then convert it to grayscale
image = cv2.imread('C:\\Users\\Sabakat\\Desktop\\images\\Trump.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Our classifier returns the ROI of the detected face as a tuple
# It stores the top left coordinate and the bottom right coordiantes
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No faces found")

# We iterate through our faces array and draw a rectangle
# over each face in faces
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# ### Let's combine face and eye detection

# In[1]:


import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('C:\\Users\\Sabakat\\Desktop\\Haarcascades\\haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('C:\\Users\\Sabakat\\Desktop\\Haarcascades\\haarcascade_eye.xml')

img = cv2.imread('C:\\Users\\Sabakat\\Desktop\\images\\essan4.png')
#img = cv2.imread('C:\\Users\\Sabakat\\Desktop\\images\\Trump.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No Face Found")

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)

cv2.destroyAllWindows()


# ### Let's make a live face & eye detection, keeping the face inview at all times

# In[1]:


import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('C:\\Users\\Sabakat\\Desktop\\Haarcascades\\haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('C:\\Users\\Sabakat\\Desktop\\Haarcascades\\/haarcascade_eye.xml')

def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img

    for (x,y,w,h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

    roi_color = cv2.flip(roi_color,1)
    return roi_color

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    cv2.imshow('Our Face Extractor', face_detector(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()


# ### Tuning Cascade Classifiers
#
# *ourClassifier*.**detectMultiScale**(input image, **Scale Factor** , **Min Neighbors**)
#
# - **Scale Factor**
# Specifies how much we reduce the image size each time we scale. E.g. in face detection we typically use 1.3. This means we reduce the image by 30% each time it’s scaled. Smaller values, like 1.05 will take longer to compute, but will increase the rate of detection.
#
#
#
# - **Min Neighbors**
# Specifies the number of neighbors each potential window should have in order to consider it a positive detection. Typically set between 3-6.
# It acts as sensitivity setting, low values will sometimes detect multiples faces over a single face. High values will ensure less false positives, but you may miss some faces.
#

# ## 2. Mini Project # 6 - Car & Pedestrian Detection
#
# **NOTE**
# - If no video loads after running code, you may need to copy your *opencv_ffmpeg.dll*
# - From: *C:\opencv2413\opencv\sources\3rdparty\ffmpeg*
# - To: Where your python is installed e.g. *C:\Anaconda2\* \
# - Once it's copied you'll need to rename the file according to the version of OpenCV you're using.
# - e.g. if you're using OpenCV 2.4.13 then rename the file as:
# - **opencv_ffmpeg2413_64.dll** or opencv_ffmpeg2413.dll (if you're using an X86 machine)
# - **opencv_ffmpeg310_64.dll** or opencv_ffmpeg310.dll (if you're using an X86 machine)
#
# To find out where you python.exe is installed, just run these two lines of code:

# In[1]:


import sys
print(sys.executable)


# In[2]:


import cv2
import numpy as np

# Create our body classifier
body_classifier = cv2.CascadeClassifier('C:\\Users\\Sabakat\\Desktop\\Haarcascades\\haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('C:\\Users\\Sabakat\\Desktop\\images\\walking.avi')

# Loop once video is successfully loaded
while cap.isOpened():

    # Read first frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()


# ### Car Detection

# In[ ]:


import cv2
import time
import numpy as np

# Create our body classifier
car_classifier = cv2.CascadeClassifier('C:\\Users\\Sabakat\\Desktop\\Haarcascades\\haarcascade_car.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('C:\\Users\\Sabakat\\Desktop\\images\\cars.avi')


# Loop once video is successfully loaded
while cap.isOpened():
    time.sleep(.05)
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)

    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()


# - **Full Body / Pedestrian Classifier ** - https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_fullbody.xml
# - **Car Classifier ** - http://www.codeforge.com/read/241845/cars3.xml__html
#
