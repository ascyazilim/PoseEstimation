# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:29:40 2022

@author: ayhant
"""

#see detailed information on here
#https://www.tensorflow.org/lite/examples/pose_estimation/overview

#import tflite_runtime.interpreter as tflite
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import cv2
import numpy as np
import time



# interpreter = tflite.Interpreter("movenet_single_pose_thunder.tflite")

# interpreter.allocate_tensors()
# output = interpreter.get_output_details()[0]


#from https://github.com/tensorflow/hub/blob/master/examples/colab/movenet.ipynb




# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
(0, 1): 'm',
(0, 2): 'c',
(1, 3): 'm',
(2, 4): 'c',
(0, 5): 'm',
(0, 6): 'c',
(5, 7): 'm',
(7, 9): 'm',
(6, 8): 'c',
(8, 10): 'c',
(5, 6): 'y',
(5, 11): 'm',
(6, 12): 'c',
(11, 12): 'y',
(11, 13): 'm',
(13, 15): 'm',
(12, 14): 'c',
(14, 16): 'c'
}

KEYPOINT_DICT = {
'nose': 0,
'left_eye': 1,
'right_eye': 2,
'left_ear': 3,
'right_ear': 4,
'left_shoulder': 5,
'right_shoulder': 6,
'left_elbow': 7,
'right_elbow': 8,
'left_wrist': 9,
'right_wrist': 10,
'left_hip': 11,
'right_hip': 12,
'left_knee': 13,
'right_knee': 14,
'left_ankle': 15,
'right_ankle': 16
}



#About MoveNet thunder int8 implementation
#https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/3


# Overview
# A convolutional neural network model that runs on RGB images and predicts human joint locations of a single person. The model is designed to be run in the browser using Tensorflow.js or on devices using TF Lite in real-time, targeting movement/fitness activities. This variant: MoveNet.SinglePose.Thunder is a higher capacity model (compared to MoveNet.SinglePose.Lightning) that performs better prediction quality while still achieving real-time (>30FPS) speed. Naturally, thunder will lag behind the lightning, but it will pack a punch.

# Model Specifications
# The following sessions describe the general model information. Please see the model card for more detailed information and quantitative analysis.

# Model Architecture
# MobileNetV2 image feature extractor with Feature Pyramid Network decoder (to stride of 4) followed by CenterNet prediction heads with custom post-processing logics. Thunder uses depth multiplier 1.75.

# Inputs
# A frame of video or an image, represented as an float32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].

# Outputs
# A float32 tensor of shape [1, 1, 17, 3].

# The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints (in the order of: [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]).

# The third channel of the last dimension represents the prediction confidence scores of each keypoint, also in the range [0.0, 1.0].



 # #Usage
 # image_path = 'image/ayhan_sit.jpg'
 # img = cv2.imread(image_path)
 # im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

 # im_rgb = cv2.resize(im_rgb, (256,256),interpolation = cv2.INTER_AREA)
 #  cv2.imshow("frame",im_rgb)
 #  cv2.waitKey(0) 
  
# # #closing all open windows 
#   cv2.destroyAllWindows()

 # data=im_rgb.copy()
 # data=np.expand_dims(data,axis=0)
 # data=data.astype("float32")


# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# interpreter = tf.lite.Interpreter(model_path="movenet_thunder_3.tflite")
# interpreter.allocate_tensors()


# interpreter.set_tensor(input_details[0]['index'], data)
# interpreter.invoke()
# keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

# points=keypoints_with_scores[0][0]*256# points=points[:,:2]
# input_data=data[0]

# keypoint=7
# im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
# for p in points:
# image = cv2.circle(im_rgb, (int(p[1]),int(p[0])), radius=0, color=(0, 0, 255),thickness=8) 

# cv2.imshow("frame",image)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
 
 
# creating the videocapture object
# and reading from the input file
# Change it to 0 if reading from webcam
 

interpreter = tf.lite.Interpreter(model_path="movenet_thunder_3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
score = 0



# Reading the video file until finished
while(cap.isOpened()):
 
# Capture frame-by-frame
 
    ret, frame = cap.read()
     
    # if video finished or no Video Input
    if not ret:
        break
     
    # time when we finish processing for this frame
    new_frame_time = time.time() 
    im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_rgb = cv2.resize(im_rgb, (256,256),interpolation = cv2.INTER_AREA)
    data=im_rgb.copy()
    data=np.expand_dims(data,axis=0)
    data=data.astype("float32")
    
    
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    points=keypoints_with_scores[0][0]*256
    points=points[:,:2]
    input_data=data[0]
    keypoint=7
    im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
    p=points[0]
    im_rgb = cv2.circle(im_rgb, (int(p[1]),int(p[0])), radius=1, color=(0, 0, 255),thickness=8)
    
   
    Elb = points[7]
    im_rgb = cv2.circle(im_rgb, (int(Elb[1]),int(Elb[0])), radius=1, color=(0, 0, 255),thickness=8)
    # print (type(points))
    
    
    #rect = (x1,y1),(x2,y2)
    rect = cv2.rectangle(im_rgb, (75,75), (175,175), color=(255,0,0))
    if (int(points.item(1)) > 75 and int(points.item(1)) <175) and int((points.item(0)) > 75 and int(points.item(0)) < 175):
        #cv2.putText(im_rgb, "Calibration ok", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if int(points.item(1)) < 120 and (int(points.item(0)) > 130 and int(points.item(0) < 150)):     #if points.item(1) < 120:
            cv2.putText(im_rgb, "right", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1)
            # score +=1
            # print(score)
            if int(points.item(1)) > 155:
                cv2.putText(im_rgb, "left", (20,20), cv2.FONT_HERSHEY_SIMPLEX())
                score +=1
                print(score)

        elif int(points.item(1)) >150 and (int(points.item(0)) > 130 and int(points.item(0) < 155)):
            cv2.putText(im_rgb, "left", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1)
            # score +=1
            # print(score)
            
            
    else:
        cv2.putText(im_rgb, "Calibration rejected", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
    #x 132 145   y 128 142 
    sonuc = cv2.putText(im_rgb,f"score: {str(score)}",(10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    cv2.putText(im_rgb, str(points), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    
    
    # for p in points:
    #     im_rgb = cv2.circle(im_rgb, (int(p[1]),int(p[0])), radius=1, color=(0, 255, 255),thickness=8) 
        
    # fps = 1/(new_frame_time-prev_frame_time)
    # #print(new_frame_time-prev_frame_time)
    # prev_frame_time = new_frame_time
        
    # converting the fps into integer
    # fps = int(fps)
    # # converting the fps to string so that we can display it on frame
    # # by using putText function
    # fps = str(fps)
    # putting the FPS count on the frame
    #cv2.putText(im_rgb, fps, (7, 70), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 3, cv2.LINE_AA)
    # displaying the frame with fps
    cv2.imshow('frame', im_rgb)
    
    # press 'Q' if you want to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv2.destroyAllWindows()


