from yolov5_tflite_inference import yolov5_tflite
import argparse
import cv2
import time
from pathlib import Path
import sys
import json
import os
import time
import board
import busio
import math
#import adafruit_mlx90640
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 
from scipy import ndimage
from scipy.interpolate import interp1d
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

import logging
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'
import csv
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from PIL import Image, ImageOps
import numpy as np
from utils2 import letterbox_image, scale_coords
from trackers.multi_tracker_zoo import create_tracker
#Create DEEPSORT Trackers
from matplotlib import cm
import subprocess
import re
import pyrealsense2 as rs
def configureCamera():
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    depth_sensor = device.first_depth_sensor()
    ## Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    advanced_mode = rs.rs400_advanced_mode(device)
    json_file_path ='/home/intelligentmedicine/Desktop/RealSense D435/DefaultPreset_D435.json'
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    json_string = str(contents).replace("'", '\"')

    advanced_mode.load_json(json_string)
    # Start streaming
    pipeline.start(config)
    return pipeline
def configureTracker():
    reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt'
    tracking_method='ocsort'
    device = 'cpu'
    half = False
    nr_sources = 1
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources
    return outputs,tracker,tracker_list
def configureClient(clientID):
    myMQTTClient = AWSIoTMQTTClient(clientID)
    # For TLS mutual authentication
    myMQTTClient.configureEndpoint("a2rhn7hbr9ksxp-ats.iot.us-east-1.amazonaws.com", 8883) #Provide your AWS IoT Core endpoint (Example: "abcdef12345-ats.iot.us-east-1.amazonaws.com")
    myMQTTClient.configureCredentials("/home/intelligentmedicine/Desktop/AWS_IoT/AmazonRootCA1.pem", "/home/intelligentmedicine/Desktop/AWS_IoT/private_RPI.pem.key", "/home/intelligentmedicine/Desktop/AWS_IoT/certificate_RPI.pem.crt") #Set path for Root CA and provisioning claim credentials
    myMQTTClient.configureOfflinePublishQueueing(-1)
    myMQTTClient.configureDrainingFrequency(2)
    myMQTTClient.configureConnectDisconnectTimeout(10)
    myMQTTClient.configureMQTTOperationTimeout(5)
    return myMQTTClient
def checkConnections(i2c,deviceList):
    #Check i2c
    currentAddresses = [hex(device_address) for device_address in i2c.scan()]
    #check for camera
    
    #Get connection status for i2c
    connectionStatus = {}
    for device in deviceList.keys():
        if deviceList[device]['Address'] in currentAddresses:
            deviceList[device]['Status'] = 'Connected'
        else:
            deviceList[device]['Status'] = 'Disconnected'
        connectionStatus[device] = deviceList[device]['Status']
    connectionStatus['IntelRealSense'] = 'Connected' #fix this
    return connectionStatus
def getCoordinates(x,y,depth_frame):
    HFOV = 86
    VFOV = 57
    dimensions = np.asarray(depth_frame.get_data()).shape
    width = dimensions[1]
    print(width)
    height = dimensions[0]
    hAngle = ((x-width/2)/(width/2))*(HFOV/2)
    vAngle = (((y-height/2)/(height/2))*(VFOV/2))
    print(vAngle)
    print("DISTANCE!",round(depth_frame.get_distance(x,y),2))
    yDistance = depth_frame.get_distance(x,y)*math.cos((vAngle+15)*math.pi/180)
    xDistance = depth_frame.get_distance(x,y)*math.sin(hAngle*math.pi/180)
    return xDistance,yDistance
def detect_image(weights,
                 img_size,
                 conf_thres,
                 iou_thres,
                 PATH,
                 numFrames,client
                 ):
    
    with open('class_names.txt') as f:
        names = [line.rstrip() for line in f]
    
    pipeline = configureCamera()
    cnt = 0
    outputs,tracker,tracker_list = configureTracker()
    
    
    
    
    ''' Setup variables needed and create csv file'''
    #Camera FOV parameters
    header = ['timestamp', 'occupantID', 'posList']
    csv_path = '/home/intelligentmedicine/Desktop/RealSense_Exports/UrbanCoworks_realtime3_15deg.csv'
    video_path = '/home/intelligentmedicine/Desktop/RealSense_Exports/UrbanCoworks_realtime3_15deg.avi'
    with open(csv_path,'w',\
              encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
    #Tracking parameters
    personDetectedLast = False #Assume nobody is in the room yet
    
    #Setup Device Dictionary
    i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

    availableDevices = ['MLX90640','BME280','VEML7700','SGP30','IntelRealSense']
    deviceList = {'MLX90640': {'Address':'0x33','Status': 'Disconnected', 'mqttClient': None},
                   'BME280': {'Address':'0x77','Status': 'Disconnected', 'mqttClient': None},
                   'VEML7700': {'Address':'0x10','Status': 'Disconnected', 'mqttClient': None},
                   'SGP30': {'Address':'0x58','Status': 'Disconnected', 'mqttClient': None},
                  'RealSenseD435': {'Address': 'USB', 'Status' : 'Disconnected','mqttClient': None}
                   }
    ''' Setup EDGE client '''
    checkConnections(i2c,deviceList)
    edgeClient = configureClient('CirculateEDGE')
    edgeClient.connect()
    edgeClient.publish(topic = 'Circulate Edge', QoS = 1, payload = 'Edge Device Connected...')
    edge_start = time.time()
    ''' Initialize cv2 video'''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path,fourcc,1.77,(640,480))
    start_time = time.time()

    while True:
        loop_start = time.time()
        #Set-up a timer for EDGE
        #Initialize EDGE client
        ''' Every 10 seconds, publish status of devices'''
        if time.time()-edge_start > 10:
            connectionStatus = checkConnections(i2c,deviceList)
            payloadMsg = json.dumps(connectionStatus)
            edgeClient.publish(topic = 'Circulate Edge', QoS = 1, payload = payloadMsg)
            edge_start = time.time()

        '''Get frame data'''        
        frames = pipeline.wait_for_frames()
        spat_filter = rs.spatial_filter()
        spat_filter.set_option(rs.option.holes_fill,2)
        depth_frame = spat_filter.process(frames)
        depth_frame = frames.get_depth_frame()
        
        
        
        if not depth_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
        colorizer.set_option(rs.option.max_distance, 5)
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_colormap_dim = depth_colormap.shape
        
        # for yolov5
        depth_colormap = depth_colormap[:,:,::-1] #flip to RGB
        image = Image.fromarray(depth_colormap)
        original_size = image.size[:2]
        size = (img_size,img_size)
        image_resized = letterbox_image(image,size)
        img = np.asarray(image)
        
        orig_img = np.asarray(image)

        #image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image_resized)
        normalized_image_array = image_array.astype(np.float32) / 255.0
        
        cnt += 1
        ''' Deep Learning'''
        yolov5_tflite_obj = yolov5_tflite(weights,img_size,conf_thres,iou_thres)

        result_boxes, result_scores, result_classes,result_class_names = yolov5_tflite_obj.detect(normalized_image_array)

        if len(result_boxes) > 0:
            #Update tracker list
            result_boxes = np.array(result_boxes)
            result_scores = np.array([result_scores]).T
            result_class_names = np.array(result_classes)
            det = np.concatenate([result_boxes,result_scores,result_class_names],axis = 1)
            outputs[0] = tracker_list[0].update(det, image_array)

            id_list = []
            cls_list = []
            conf_list = []
            box_list = []                
            # fontScale 
            fontScale = 0.5
                
            # Blue color in BGR 
            color = (255, 255, 0) 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            # Line thickness of 1 px 
            thickness = 1
            #Tracker info and plotting
            if len(outputs[0]) > 0:
                pos_list = []
                for j, (output, conf) in enumerate(zip(outputs[0], det[:,:4])):
                    bboxes = output[0:4]
                    bboxes = scale_coords(size,np.array(bboxes),(original_size[1],original_size[0]))
                    bboxes = bboxes[0]
                    box_list.append(bboxes)
                    id = int(output[4])
                    id_list.append(id)
                    cls = output[5]
                    c = int(cls)
                    cls_list.append(int(c))
                    label = (f'{id} {names[c]}')
                    org = (int(bboxes[0]),int(bboxes[1]))
                    org_bot = (int(bboxes[0]),int(bboxes[3])+12)
                    
                    midPointX = ((int(bboxes[0])+int(bboxes[2]))//2)
                    midPointY = ((int(bboxes[1])+int(bboxes[3]))//2)
                    
                    
                    xDistance,yDistance = getCoordinates(midPointX,midPointY,depth_frame)
                    pos_list.append((round(xDistance,2),round(yDistance,2)))
                    print((round(xDistance,2),round(yDistance,2)))
                    cv2.circle(img,(midPointX,midPointY),radius = 5,color = (255,0,0),thickness = 5)
                    
                    cv2.rectangle(img, (int(bboxes[0]),int(bboxes[1])), (int(bboxes[2]),int(bboxes[3])), (255,0,0), 1)
                    cv2.putText(img, "ID:"  + str(id) +  ', ' + str(int(100*result_scores[j])) + '%  ' + str(names[c]), org, font,  
                                fontScale, color, thickness, cv2.LINE_AA)
                    cv2.putText(img,"X:" + str((int(bboxes[0])+int(bboxes[2]))//2) + "Y:" + str(int(bboxes[3])), org_bot, font,  
                                fontScale, color, thickness, cv2.LINE_AA)
                    
                    save_result_filepath ='test.jpg' #image_url.split('/')[-1].split('.')[0] + 'yolov5_output.jpg'
                    
                    #cv2.imwrite(save_result_filepath,img[:,:,::-1])
                    
            font = cv2.FONT_HERSHEY_SIMPLEX 
            end_time = time.time()
            
            #Perform checks for saving
            if 0 in np.unique(result_classes): #1 is person
                personDetected = True #person is detected
            else:
                personDetected = False #nobody is detected
            
            if personDetected:
                #Update current ID's
                peopleID = [id_list[i] for i in range(len(cls_list)) if cls_list[i] == 0]
                boxes = [box_list[i] for i in range(len(cls_list)) if cls_list[i]==0]
                data = {}
                data['timestamp'] = str(datetime.now())
                data['occupantID'] = str(peopleID)
                data['posList'] = str(pos_list[0])
                with open(csv_path,'a',\
                      encoding='UTF8', newline='') as f:
                    writer = csv.DictWriter(f,fieldnames = header)
                    # write the header
                    writer.writerow(data)
                #data['pos'] = str(pos_list)
                payloadMsg = json.dumps(data)
                client.publish(topic = "IntelRealSense", QoS=1, payload = payloadMsg )
                '''
                currTime = datetime.now()
                if len(peopleID) < len(lastPeopleID):
                    peopleLeft = list(set(lastPeopleID)-set(peopleID))
                    for id in peopleLeft:
                        print(f"Person {id} has left the room at {currTime}")
                        logger.info(f'Person {id} has left the room at frame {numFrames}')
                elif len(peopleID) > len(lastPeopleID):
                    peopleEntered = list(set(peopleID)-set(lastPeopleID))
                    for id in peopleEntered:
                        print(f"Person {id} has entered the room at {currTime}")
                        logger.info(f'Person {id} has Entered the room at frame {numFrames}')
                lastPeopleID = peopleID
                img = Image.fromarray(img)
                orig_img = Image.fromarray(orig_img)
                folder_name = PATH[48:]
                #img.save(PATH + '/Annotated/' + str(numFrames) +'_'+folder_name+ '.png')
                #orig_img.save(PATH + '/Unannotated/' + str(numFrames) +'_'+folder_name+ '.png')
                numFrames += 1
                '''
                
                
            personDetectedLast = personDetected
            '''
            if (counter_end-counter_start).seconds == 10:
                print(f"There are currently {len(peopleID)} people in the room at time {datetime.now()}")
                logger.info(f'There are currently {len(peopleID)} people in the room')
                counter_start = datetime.now()
            '''
        cv2.imshow('test',img[:,:,::-1])
        video.write(img[:,:,::-1])
        cv2.waitKey(1)
        print("FPS:", 1/(time.time()-loop_start))
        if time.time() - start_time > 30:
            cv2.destroyAllWindows()
            video.release()
            break
            
        
        #image = cv2.imread(image_url)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--weights', type=str, default='defaultrealsense_augmented3.tflite', help='model.tflite path(s)')
    parser.add_argument('-i','--img_path', type=str, required=False, help='image path')  
    parser.add_argument('--img_size', type=int, default=416, help='image size')  
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')

    
    opt = parser.parse_args()
    
    print(opt)
    detect_image(opt.weights,opt,opt.img_size,opt.conf_thres,opt.iou_thres)


    

