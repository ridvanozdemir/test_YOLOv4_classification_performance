# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:45:59 2022

@author: rıdvan özdemir
test your YOLOv4 model's classification performance'
"""
import cv2
import numpy as np
import os
import glob
import shutil
import time
import datetime



#pat your object list file's folder
path_object_list = 'D:/Yolo_v4/darknet/build/darknet/x64'
#create dictionary 
number_of_products={}
#make list of objects in txt file
labels=[]
class_list = glob.glob(os.path.join(path_object_list, 'object_list.txt'))

with open(class_list[0], 'r') as t:
    for raw in t:
        urun = raw.split(' ')[0]
        labels.append(urun)

#path of images folder 
parent_dir= "enter_your_path/class_folders/"


#create folders accodring to object name      
for directory in labels:
    path = os.path.join(parent_dir, directory[:-1])
    os.mkdir(path)
    print("Directory '% s' created" % directory)
    

folder = "input folder path"
file_path_n ="output folder path"
total_time = time.time()


TP=0
FP=0


number_of_products={}
file_names=[]
all_file=[]




#reset all
i=0 
for l in labels:
    number_of_products[l]=0



colors = ["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0","100,255,255","100,100,255","100,255,100","100,100,100",
          "100,0,255","100,0,0","100,255,0","0,100,255","100,0,100","100,100,255"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors,(18,1))

#path your cfg and weight files
model = cv2.dnn.readNetFromDarknet("D:/Yolo_v4/darknet/build/darknet/x64/cfg/yolov4.cfg","D:/Yolo_v4/darknet/build/darknet/x64/backup/yolov4.weights")
layers = model.getLayerNames()
output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers()]
k=0

for filename in glob.glob(os.path.join(folder, '*.jpg')):
#for filename in os.listdir(folder):
    img=cv2.imread(os.path.join(folder,filename))  
    txt_filename = filename[:-4]+".txt"
    
    with open(txt_filename, 'r') as f: # open in readonly mode
        for satir in f:
            sinif = satir.split(' ')[0]
    
    
    img_width = img.shape[1]
    img_height = img.shape[0]




    img_blob = cv2.dnn.blobFromImage(img, 1/255, (256,256), swapRB=True, crop=False)
    
    prev_time = time.time()
    model.setInput(img_blob)
    
    detection_layers = model.forward(output_layer)
    
    
    ############## NON-MAXIMUM SUPPRESSION - OPERATION 1 ###################
    
    ids_list = []
    boxes_list = []
    confidences_list = []
    
    ############################ END OF OPERATION 1 ########################
    
    
    
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence > 0.30:
                
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                
                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))
                
                
                ############## NON-MAXIMUM SUPPRESSION - OPERATION 2 ###################
                
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                
                ############################ END OF OPERATION 2 ########################
                
                
                
    ############## NON-MAXIMUM SUPPRESSION - OPERATION 3 ###################
                
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
         
    for max_id in max_ids:
        
        max_class_id = max_id
        box = boxes_list[max_class_id]
        
        start_x = box[0] 
        start_y = box[1] 
        box_width = box[2] 
        box_height = box[3] 
         
        predicted_id = ids_list[max_class_id]
        #label = labels[predicted_id][:-1]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]
      
    ############################ END OF OPERATION 3 ########################
                
        end_x = start_x + box_width
        end_y = start_y + box_height
                
        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]
        

        
        calc_time= (time.time() - prev_time)        
        fps = int(1/calc_time)
        #print("FPS: {}".format(fps))   
        #print(calc_time)
        number_of_products[label]=number_of_products[label]+1
        label_c = "{}: {:.2f}%".format(label, confidence*100)
        #print("predicted object {}".format(label_c))
        cv2.rectangle(img, (start_x,start_y),(end_x,end_y),box_color,3)
        cv2.putText(img,label_c,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, box_color, 3)
        dosya_adi= label[:-1] + '_' + str(confidence)[1:5] + '_' + str(calc_time)[:4] + '_' + str(k) +".jpg"

        if sinif == str(predicted_id):
            TP+=1
        else:
            FP+=1
            save_names_conf = filename[30:-4] + " " + labels[int(sinif)][:-1] + " " + label_c
            file_names.append(save_names_conf)        
        
        all_file.append(os.path.join(file_path_n,filename))
        parent_dir
        hedef_1_txt= os.path.join(parent_dir,labels[predicted_id][:-1])
        hedef_1= os.path.join(hedef_1_txt, filename)
        k+=1
        
        try:
            #move jpg files
            #shutil.move(os.path.join(folder,filename), hedef_1_txt)
            # copy txt files 
            #shutil.copy(os.path.join(folder,filename), hedef_1_txt)
            cv2.imwrite(os.path.join(hedef_1_txt, filename[33:]),img)
        except FileNotFoundError:
            print("file was not found")
        
print("total time: ", (time.time() - total_time))
print("image number: ", k)
print("TP: ", TP)
print("FP: ", FP)
print("prediction accuray: " + str(TP/(TP+FP)))
