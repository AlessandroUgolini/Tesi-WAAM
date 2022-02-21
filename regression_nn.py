# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:08:00 2022

@author: Utente
"""

#imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score,roc_curve

from create_dataset_noblack import FrameDataset

#let's define some parameters

root_dir='../Codicev2/frames'
csv_file='annotations_regression.csv'


num_classes = 3

n_frames=238078
n_frames_0 = 72583
n_frames_1= 54797
n_frames_2=110698

classes=['stable','arising','unstable']

weights=torch.tensor([1/n_frames_0,1/n_frames_1,1/n_frames_2])

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of epochs. Default value 100.", default=5)
    parser.add_argument("--learning-rate", type=float, help="Learning rate. Default value 0.001.", default=0.001)
    parser.add_argument("--batch-size", type=int, help="Batch size. Default value 10.", default=10)
    parser.add_argument("--threshold", type=float, help="Threshold for regression. Default value 0.5.", default=0.5)
    parser.add_argument("--load", type=bool, help="Indicate if the model has to be loaded or not.Default value False", default=False)
    args = parser.parse_args()
    return args

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            

def initialize_model(num_classes, feature_extract, load, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
   
    model_ft = models.resnet101(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    #if we want to turn classification into regression we need to have only one class in exit
    model_ft.fc = nn.Linear(num_ftrs, 1)
    input_size = 224
    if load:
        model_ft.load_state_dict(torch.load('model_regression.pth'))
    
    return model_ft, input_size

def train_model(model, dataloader, loss_fn, optimizer, writer, device, num_epochs, threshold, is_inception=False):
    
    i=0
    
    running_loss=0.0
    
    y_pred = []
    y_true = []
    
    num_batches=len(dataloader)
    
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device=device)
        y = y.to(device=device,dtype=torch.float)
        
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #data storing
        
        for p in pred:
            if p>1:
                p=1
            if p<0:
                p=0
        
        y_clipped_p=[]
        for p in pred:
            if p<threshold:
                y_clipped_p.append(0)
            else:
                y_clipped_p.append(1)
                
        y_pred.extend(y_clipped_p)
        
        running_loss += loss.item() # save loss
        
        y = y.data.cpu().numpy()
        y_clipped_t=[]
        for i in y:
            if p<threshold:
                y_clipped_t.append(0)
            else:
                y_clipped_t.append(1)
                
        y_true.extend(y_clipped_t) # Save truth
        
    
    running_loss=running_loss/num_batches
    score=roc_auc_score(y_true, y_pred)
    print(round(score,3))
    print(round(running_loss,3))
    writer.add_scalar('Loss/train', running_loss,(1+num_epochs) * len(dataloader))
    writer.add_scalar('ROC auc score/train', score,(1+num_epochs) * len(dataloader))

    
   
        
def test_model(model, dataloader, loss_fn, device,num_epochs, threshold):

    test_loss=0.0
    
    y_pred = []
    y_true = []

    num_batches = len(dataloader)

    with torch.no_grad():
        for X, y in dataloader:
            
            X=X.to(device=device)
            y=y.to(device=device)
            
            pred = model(X)
            loss = loss_fn(pred.squeeze(), y)
            
            test_loss += loss.item() # save loss
            
            for p in pred:
                if p>1:
                    p=1
                if p<0:
                    p=0
        
        
            y_clipped_p=[]
            for p in pred:
                if p<threshold:
                    y_clipped_p.append(0)
                else:
                    y_clipped_p.append(1)
            
            y_pred.extend(y_clipped_p)
            
            y = y.data.cpu().numpy()
            y_clipped_t=[]
            for i in y:
                if p<threshold:
                    y_clipped_t.append(0)
                else:
                    y_clipped_t.append(1)
                    
            y_true.extend(y_clipped_t) # Save truth
            
    score=roc_auc_score(y_true, y_pred)
    
    test_loss /= num_batches
    
    writer.add_scalar('Loss/valid', 
                      test_loss, 
                      (1+num_epochs) * len(dataloader))
            
    writer.add_scalar('ROC auc score/valid', 
                      score, 
                      (1+num_epochs) * len(dataloader))
    
    print("Test loss {:.3} Score {:.2}".format(test_loss,score))
    return test_loss,score

if  __name__ == '__main__':
    
    args=get_parser()
    
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate=args.learning_rate
    reg_threshold=args.threshold
    load=args.load

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    writer=SummaryWriter()
    model_ft, input_size = initialize_model(num_classes, feature_extract, load, use_pretrained=True)
    
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    
    model_ft = model_ft.to(device)
    
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate)
    
    # Print the model we just instantiated
    print(model_ft)
    
    # Setup the loss fxn
    weights=weights.to(device)
    criterion = nn.MSELoss()
    
    print("Initializing Datasets and Dataloaders...")
    
    dataset= FrameDataset(csv_file=csv_file, root_dir=root_dir)
    
    nframes=len(dataset)
    print(nframes)
    
    spl1=int(nframes/100*80)
    spl2=nframes-spl1
    
    print(spl1)
    print(spl2)
    
    # Create training and validation datasets
    train_set,test_set=torch.utils.data.random_split(dataset,[spl1,spl2])
    
    # Create training and validation dataloaders
    train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)
    
    # Train and evaluate
    for t in range(num_epochs):
        print(f"Epoch {t+1}/"+str(num_epochs)+"\n-------------------------------")
        train_model(model_ft, train_loader, criterion, optimizer_ft, writer, device, t, reg_threshold)
        tl,sc= test_model(model_ft, test_loader, criterion, device,t, reg_threshold )
    print("Done!")
    
    model_ft=model_ft.to(device='cpu')
    torch.save(model_ft.state_dict(), 'model_regression.pth')
    writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, 'threshold': reg_threshold},{'hparam/ROC auc score': sc, 'hparam/loss': tl})
    
    writer.close()