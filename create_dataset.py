# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:59:48 2021

@author: Utente
"""

import cv2
import os
import pandas as pd
import numpy as np
import skimage as io
import csv

import torch
from torch.utils.data import Dataset



def get_frames(videopath,durations):
    for i in range(1,len(videopath)):
            
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
                    #per ogni frame salvo una immagine
                    cv2.imwrite("frames/vid"+str(i)+" "+str(framecount)+".jpg", frame)
                    framecount+=1
                # Break the loop
                else: 
                     break
            
            # When everything done, release the video capture object
            cap.release()
            

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
        
        if self.trasform:
            image=self.trasform(image)
        
        return (image,ylabel)

  
if __name__ == '__main__':
    
    videos=[]
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_3m-min 2.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_3m-min.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_3m-min_1 2.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_3m-min_1.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_5m-min 2.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_5m-min.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_5m-min_1 2.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_20mm_filo_5m-min_1.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_30mm.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_40mm.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_grande_20mm.h264')
    videos.append('VIDEO_MONITORAGGIO/video_iso200_ss1500_irideaperta_cilindro_piccolo_20mm.h264')
    
    durations=[15620,15700,15700,15750,15750,15827,15827,15794,15794,20847,31266,31204,15750]
    
    threshold=np.zeros((13,4)).astype(int)
    
    threshold[0]=[42,3300,3990,15619]
    threshold[1]=[114,6000,9600,15619]
    threshold[2]=[114,6750,13200,15619]
    threshold[3]=[170,8190,13500,15678]
    threshold[4]=[171,8190,12900,15672]
    threshold[5]=[103,4800,5550,15667]
    threshold[6]=[103,4800,5550,15667]
    threshold[7]=[93,4890,5700,15650]
    threshold[8]=[93,4890,5610,15650]
    threshold[9]=[79,4440,6000,20764]
    threshold[10]=[46,7140,20400,31017]
    threshold[11]=[116,6510,20400,31090]
    threshold[12]=[63,3990,4980,15660]
    
    create_csv(threshold,durations)
    

    
