# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:09:17 2022

@author: Utente
"""

import os
import pandas as pd
import numpy as np
from skimage import io
import csv

import torch
from torch.utils.data import Dataset
from torchvision import transforms


            

def create_csv(threshold,durations):
    filename = "annotations_regression.csv"
    
    # writing to csv file 
    with open(filename, 'w',newline="") as csvfile: 
        
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        
        #iterate over the 13 videos
        for i in range(len(durations)):
            
            start=threshold[i][0]
            rise=threshold[i][1]
            unstable=threshold[i][2]
            
            #iterate over the video frames
            for j in range(durations[i]):
                
                if j >= start:
                           
                    name="vid"+str(i)+" "+str(j)+".jpg"
                    
                    label=0.000
                    if j>=rise and j<unstable:
                        label=(j-rise)/(unstable-rise)
                        label=round(label,4)
                        print(label)
                    if j>=unstable:
                        label=1.000
                        
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
    
    create_csv(threshold,durations)
    
    