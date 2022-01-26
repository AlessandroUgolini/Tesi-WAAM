# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 12:11:09 2022

@author: Utente
"""

import cv2
import os
import pandas as pd
import numpy as np
from skimage import io
import csv

import torch
from torch.utils.data import Dataset
from torchvision import transforms



def get_frames(videopath, durations, threshold):
    filename = "annotations.csv"
    with open(filename, 'w',newline="") as csvfile:
        
        csvwriter = csv.writer(csvfile)
        
        for i in range(len(videopath)):
                
                print("video"+str(i))
                
                cap = cv2.VideoCapture(videopath[i])
                
                if (cap.isOpened()== False):
                    print("Error opening video stream or file")
                    
                framecount=0
                #iterate over the video frame
                while cap.isOpened() and framecount<durations[i]:
                    
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    print (framecount)
                    if ret == True:
                        if framecount >= threshold[i][0]:
                            #per ogni frame salvo una immagine
                            cv2.imwrite("frames/vid"+str(i)+" "+str(framecount)+".jpg", frame)
                            
                            #let's build the couple name-label
                            name="vid"+str(i)+" "+str(framecount)+".jpg"
                            
                            label=0
                            if framecount>=threshold[i][1] and framecount<threshold[i][2]:
                                label=1
                            if framecount>=threshold[i][2]:
                                label=2
                                
                            row=(name,label)
                            # writing the couple in the csv
                            csvwriter.writerow(row)
                            
                        framecount+=1
                    # Break the loop
                    else: 
                         break
                
                # When everything done, release the video capture object
                cap.release()
        
        csvfile.close()
            

def create_csv(threshold,durations):
    filename = "annotations.csv"
    
    # writing to csv file 
    with open(filename, 'w',newline="") as csvfile: 
        
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        
        #iterate over the 13 videos
        for i in range(len(durations)):
            #iterate over the video frames
            for j in range(durations[i]):
                #now we build the copule name label
                name="vid"+str(i)+" "+str(j)+".jpg"
                label=0
                if j>=threshold[i][0] and j<threshold[i][1]:
                    label=1
                if j>=threshold[i][1] and j<threshold[i][2]:
                    label=2
                if j>=threshold[i][2] and j<threshold[i][3]:
                    label=3
                if j>=threshold[i][3]:
                    label=0
                row=(name,label)
                # writing the couple in the csv
                csvwriter.writerow(row)
        csvfile.close()
        
    

class FrameDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
        
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        
        img_path=os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image=io.imread(img_path)
        ylabel=torch.tensor(int(self.annotations.iloc[index,1])) 
        
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        image=preprocess(image)
        
        return (image,ylabel)

  
if __name__ == '__main__':
    
    videos=[]
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_3m-min 2.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_3m-min.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_3m-min_1 2.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_3m-min_1.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_5m-min 2.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_5m-min.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_5m-min_1 2.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_5m-min_1.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_30mm.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_40mm.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_grande_20mm.h264')
    videos.append('../codice/VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_piccolo_20mm.h264')
    
    
    threshold=np.zeros((13,3)).astype(int)
    
    threshold[0]=[42,3300,3990]
    threshold[1]=[114,6000,9600]
    threshold[2]=[114,6750,13200]
    threshold[3]=[170,8190,13500]
    threshold[4]=[171,8190,12900]
    threshold[5]=[103,4800,5550]
    threshold[6]=[103,4800,5550]
    threshold[7]=[93,4890,5700]
    threshold[8]=[93,4890,5610]
    threshold[9]=[79,4440,6000]
    threshold[10]=[46,7140,20400]
    threshold[11]=[116,6510,20400]
    threshold[12]=[63,3990,4980]
    
    durations=[15620,15620,15620,15679,15673,15668,15668,15651,15651,20765,31018,31091,15661]
    
    get_frames(videos, durations, threshold)
    
    #create_csv(threshold,durations)
    