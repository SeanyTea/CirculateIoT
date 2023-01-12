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
import adafruit_mlx90640
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 
from scipy import ndimage
from scipy.interpolate import interp1d
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

def convert_data(frame):
    mlx_shape = (24,32)
    mlx_interp_val = 10 # interpolate # on each dimension
    data_array = (np.reshape(frame,mlx_shape))*9/5+32 #Convert to F
    plt.imshow(data_array)
    data_array = np.clip(data_array,75,np.max(data_array))
    data_array = ndimage.zoom(data_array,10)
    my_cm = plt.cm.get_cmap('rainbow')
    normed_data = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))
    #mapped_data = my_cm(data_array)
    mapped_data = np.uint8(cm.rainbow(normed_data)*255)#Image.fromarray(np.uint8(cm.rainbow(normed_data)*255))
    mapped_data = mapped_data[:,:,:3] #Convert from RGBA to RGB

    im = Image.fromarray(mapped_data)
    return im
def convert_to_meters(xLoc,yLoc):
    #Maps pixels to meters. Current setup is minX -55 , minY - 71, maxX - 200, maxY - 240
    #101 inches in x direction = 2.5656 m
    #left is -1.1938 right is 1.3716 m
    #back is 2.921 m, front is 0 m
    xInterp = interp1d([60,240],[-1.1938,1.3716])
    yInterp = interp1d([75,240],[2.921,0])
    x_m = xInterp(xLoc)
    y_m = yInterp(yLoc)
    print(xLoc)
    return float(x_m),float(y_m)
    
    
def detect_image(weights,
                 img_size,
                 conf_thres,
                 iou_thres,
                 PATH,
                 numFrames,client
                 ):
    with open('class_names.txt') as f:
        names = [line.rstrip() for line in f]
    
        #Setup MLX90640
        # initialize i2c bus
        # Configure depth and color streams
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
    '''
    preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
    print('preset range:'+str(preset_range))
    for i in range(int(preset_range.max)):
        visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
        print('%02d: %s' %(i,visulpreset))
        if visulpreset == "High Accuracy":
            depth_sensor.set_option(rs.option.visual_preset, i)
    '''
    advanced_mode = rs.rs400_advanced_mode(device)
    json_file_path ='/home/intelligentmedicine/Desktop/RealSense D435/DefaultPreset_D435.json'
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    json_string = str(contents).replace("'", '\"')

    advanced_mode.load_json(json_string)
    # Start streaming
    pipeline.start(config)
    #depth_sensor = device.first_depth_sensor()
    cnt = 0
    #Setup tracker
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



    personDetectedLast = False #Assume nobody is in the room yet
    counter_start = datetime.now()
    been_disconnected = 0
    while True:
        start_time = time.time()

        #Obtain mlx frame
        
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
        colorizer.set_option(rs.option.min_distance, 0)
        colorizer.set_option(rs.option.max_distance, 6)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        '''
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        '''
        # Show images
        
        #cv2.imwrite(savePath + str(cnt) + '.png',depth_colormap)
        cnt += 1
        cv2.waitKey(1)
        depth_colormap = depth_colormap[:,:,::-1]
        image = Image.fromarray(depth_colormap)
        original_size = image.size[:2]
        size = (img_size,img_size)
        image_resized = letterbox_image(image,size)
        img = np.asarray(image)
        #cv2.imshow('RealSense', img[:,:,::-1])

        orig_img = np.asarray(image)
        
        #image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image_resized)
        print(image_array.shape)
        normalized_image_array = image_array.astype(np.float32) / 255.0
        yolov5_tflite_obj = yolov5_tflite(weights,img_size,conf_thres,iou_thres)

        result_boxes, result_scores, result_classes,result_class_names = yolov5_tflite_obj.detect(normalized_image_array)

        if len(result_boxes) > 0:
            personDetected = False
            #Update tracker list
            result_boxes = np.array(result_boxes)
            result_scores = np.array([result_scores]).T
            result_class_names = np.array(result_classes)
            det = np.concatenate([result_boxes,result_scores,result_class_names],axis = 1)
            #start_time = time.time()
            outputs[0] = tracker_list[0].update(det, image_array)
            #print(time.time()-start_time)
            #outputs[0] = []
            id_list = []
            cls_list = []
            conf_list = []
            box_list = []
            org = (20, 40) 
                
            # fontScale 
            fontScale = 0.5
                
            # Blue color in BGR 
            color = (255, 255, 0) 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            # Line thickness of 1 px 
            thickness = 1
            #Tracker info and plotting
            if len(outputs[0]) > 0:
                for j, (output, conf) in enumerate(zip(outputs[0], det[:,:4])):
                    bboxes = output[0:4]
                    bboxes = scale_coords(size,np.array(bboxes),(original_size[1],original_size[0]))
                    box_list.append(bboxes[0])
                    id = int(output[4])
                    id_list.append(id)
                    cls = output[5]
                    c = int(cls)
                    cls_list.append(int(c))
                    label = (f'{id} {names[c]}')
                    
                    x_coord = ((int(bboxes[0][0])+int(bboxes[0][2]))//2)
                    y_coord = int(bboxes[0][3])
                    org = (int(bboxes[0][0]),int(bboxes[0][1]))
                    org_bot = (int(bboxes[0][0]),int(bboxes[0,3])+12)
                    center_of_mass_x = ((int(bboxes[0][0])+int(bboxes[0][2]))//2)
                    center_of_mass_y = ((int(bboxes[0][1])+int(bboxes[0][3]))//2)
                    print("DEPTH",depth_frame.get_distance(center_of_mass_x,center_of_mass_y))
                    print("DEPTH2",depth_frame.get_distance(center_of_mass_x,int(bboxes[0][3])))

                    cv2.circle(img,(center_of_mass_x,center_of_mass_y),radius = 5,color = (255,0,0),thickness = 5)
                    
                    cv2.rectangle(img, (int(bboxes[0][0]),int(bboxes[0][1])), (int(bboxes[0][2]),int(bboxes[0][3])), (255,0,0), 1)
                    cv2.putText(img, "ID:"  + str(id) +  ', ' + str(int(100*result_scores[j])) + '%  ' + str(names[c]), org, font,  
                                fontScale, color, thickness, cv2.LINE_AA)
                    cv2.putText(img,"X:" + str((int(bboxes[0][0])+int(bboxes[0][2]))//2) + "Y:" + str(int(bboxes[0][3])), org_bot, font,  
                                fontScale, color, thickness, cv2.LINE_AA)
                    
                    save_result_filepath ='test.jpg' #image_url.split('/')[-1].split('.')[0] + 'yolov5_output.jpg'
                    cv2.imshow('test',img[:,:,::-1])
                    cv2.waitKey(1)
                    #cv2.imwrite(save_result_filepath,img[:,:,::-1])
                    
            font = cv2.FONT_HERSHEY_SIMPLEX 
            end_time = time.time()
            #Perform checks for saving
            print(cls_list)
            if 0 in np.unique(result_classes): #1 is person
                personDetected = True #person is detected
            else:
                personDetected = False #nobody is detected
            
            if personDetected and personDetectedLast ==False:
                currVideoTime = datetime.now()
                peopleID = [id_list[i] for i in range(len(cls_list)) if cls_list[i] == 0]
                boxes = [box_list[i] for i in range(len(cls_list)) if cls_list[i]==0]
                lastPeopleID = peopleID #to detect changes
                currTime = datetime.now()
                for id in peopleID:
                    print(f"Person {id} has entered the room at {currTime}")
                if PATH == 'False':
                    time_str = datetime.now().strftime("%m_%d_%Y %H:%M:%S")
                    name_list = time_str.split(' ')
                    folder_name = name_list[0] + '_'+name_list[1]
                    PATH = "/home/intelligentmedicine/Desktop/CirculateLogs/" + folder_name
                    os.makedirs(PATH)
                    os.makedirs(PATH + "/Annotated")
                    os.makedirs(PATH + "/Unannotated")
                    numFrames = 0
                logging.basicConfig(filename = PATH + '/std.log', force = True,format = '%(asctime)s %(message)s',filemode = 'w')
                logger = logging.getLogger()
                logger.setLevel(logging.DEBUG)
            if personDetected and personDetectedLast is personDetected:
                #Update current ID's
                peopleID = [id_list[i] for i in range(len(cls_list)) if cls_list[i] == 0]
                boxes = [box_list[i] for i in range(len(cls_list)) if cls_list[i]==0]
                pos_list = []
                data = {}
                '''
                for i in range(len(peopleID)):
                    x_m,y_m = convert_to_meters(int((boxes[i][0]+boxes[i][2])/2),boxes[i][3])
                    pos_list.append((round(x_m,2),round(y_m,2)))
                    #data[names[cls_list[i]] = 
                    #client.publish(topic = "MLX90640", QoS=1, payload = paylodmsg_json )
                '''
                data['timestamp'] = str(datetime.now())
                data['occupantID'] = str(peopleID)
                #data['pos'] = str(pos_list)
                payloadMsg = json.dumps(data)
                client.publish(topic = "IntelRealSense", QoS=1, payload = payloadMsg )

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
                
                

            if personDetected == False and personDetectedLast == True:
                print(f"Nobody present in the room at frame {numFrames}")
                logging.shutdown()
                counter_end = datetime.now()
                with open('readme.txt','w') as f:
                    f.write(f"Weights used: {weights}\n")
                    f.write(f"Total number of frames: {numFrames}\n")
                    #f.write(f"Total number of seconds: {(counter_end-currVideoTime).seconds}\n" )
                    #f.write(f"Average FPS: {(numFrames/(counter_end-currVideoTime).seconds)}\n")
                counter_start = counter_start.now()
            personDetectedLast = personDetected
            counter_end = datetime.now()
            if (counter_end-counter_start).seconds == 10:
                print(f"There are currently {len(peopleID)} people in the room at time {datetime.now()}")
                logger.info(f'There are currently {len(peopleID)} people in the room')
                counter_start = datetime.now()
        print("FPS:", 1/(time.time()-start_time))

            
        
        #image = cv2.imread(image_url)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--weights', type=str, default='yolov5s-fp16.tflite', help='model.tflite path(s)')
    parser.add_argument('-i','--img_path', type=str, required=False, help='image path')  
    parser.add_argument('--img_size', type=int, default=416, help='image size')  
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')

    
    opt = parser.parse_args()
    
    print(opt)
    detect_image(opt.weights,opt,opt.img_size,opt.conf_thres,opt.iou_thres)


    

