from realsense_segmentation import detect_image
import argparse
from glob import glob
import os

import time
import board
import busio
import adafruit_mlx90640
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 
from scipy import ndimage
import cv2
#DeepSort
import argparse
import scipy.misc
import matplotlib.image
from PIL import Image
from matplotlib import cm

from pathlib import Path
import sys

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from trackers.multi_tracker_zoo import create_tracker
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import time
def helloworld(self,params,packets):
    print("Received message from IoT core")
    print("Topic:", packets.topic)
    print("Payload:", packets.payload)
# For certificate based connection
myMQTTClient = AWSIoTMQTTClient("RealSense")
# For TLS mutual authentication
myMQTTClient.configureEndpoint("a2rhn7hbr9ksxp-ats.iot.us-east-1.amazonaws.com", 8883) #Provide your AWS IoT Core endpoint (Example: "abcdef12345-ats.iot.us-east-1.amazonaws.com")
myMQTTClient.configureCredentials("/home/intelligentmedicine/Desktop/AWS_IoT/AmazonRootCA1.pem", "/home/intelligentmedicine/Desktop/AWS_IoT/private_RPI.pem.key", "/home/intelligentmedicine/Desktop/AWS_IoT/certificate_RPI.pem.crt") #Set path for Root CA and provisioning claim credentials
myMQTTClient.configureOfflinePublishQueueing(-1)
myMQTTClient.configureDrainingFrequency(2)
myMQTTClient.configureConnectDisconnectTimeout(10)
myMQTTClient.configureMQTTOperationTimeout(5)

myMQTTClient.connect()
 
def detect_from_MLX(weights,img_size,conf_thres,iou_thres,PATH,cnt,client):
    detect_image(weights,img_size,conf_thres,iou_thres,PATH,cnt,client)
def convert_data(frame):
    data_array = (np.reshape(frame,mlx_shape))*9/5+32 #Convert to F
    plt.imshow(data_array)
    data_array = np.clip(data_array,78,np.max(data_array))
    data_array = ndimage.zoom(data_array,10)
    my_cm = plt.cm.get_cmap('rainbow')
    normed_data = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))
    #mapped_data = my_cm(data_array)
    mapped_data = np.uint8(cm.rainbow(normed_data)*255)#Image.fromarray(np.uint8(cm.rainbow(normed_data)*255))
    mapped_data = mapped_data[:,:,:3] #Convert from RGBA to RGB

    im = Image.fromarray(mapped_data)
    return im




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--weights', type=str, default='yolov5s-fp16.tflite', help='model.tflite path(s)')
    parser.add_argument('-s','--save_path', type=str,default = 'False', help='folder path')  
    parser.add_argument('--img_size', type=int, default=416, help='image size') 
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')

    
    opt = parser.parse_args()
    PATH = opt.save_path
    if opt.save_path == 'True':
        time_str = datetime.now().strftime("%m_%d_%Y %H:%M:%S")
        name_list = time_str.split(' ')
        folder_name = name_list[0] + '_'+name_list[1]
        PATH = "/home/intelligentmedicine/Desktop/CirculateLogs/" + folder_name
        os.makedirs(PATH + "/Annotated")
        os.makedirs(PATH + "/Unannotated")
    cnt = 0
    detect_from_MLX(opt.weights,opt.img_size,opt.conf_thres,opt.iou_thres,PATH,cnt,myMQTTClient)
        #print(time.time()-start_time)
        #print('FPS:',1/(end_time-start_time))

            

        
    #detect_from_folder_of_images(opt.weights,opt.folder_path,opt.img_size,opt.conf_thres,opt.iou_thres)
