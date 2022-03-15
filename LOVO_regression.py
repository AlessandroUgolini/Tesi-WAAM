# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 17:01:59 2022

@author: Utente
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score,roc_curve

from create_dataset_regression import FrameDataset

#let's define some parameters

root_dir='../Codicev2/frames'
csv_file='../Regressione/annotations_regression.csv'


num_classes = 3

nvideo=13

n_frames=238078
n_frames_0 = 72583
n_frames_1= 54797
n_frames_2=110698

classes=['stable','arising','unstable']

video_index=[0,15579,31085,46591,62100,77602,93167,108732,124290,139848,160534,191506,222480,238077]

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
    parser.add_argument("--test-set", type=int, help="Test set. Default value 1.", default=1)
    args = parser.parse_args()
    return args

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
   
    model_ft = models.resnet101(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    #if we want to turn classification into regression we need to have only one class in exit
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    input_size = 224
    
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
        
        y_pred.extend(pred)
        
        running_loss += loss.item() # save loss
        
        y = y.data.cpu().numpy()
        y_clipped_t=[]
        for i in y:
            if i<threshold:
                y_clipped_t.append(0)
            else:
                y_clipped_t.append(1)
                
        y_true.extend(y_clipped_t) # Save truth
        
    
    y_t = torch.FloatTensor(y_true)
    y_p = torch.FloatTensor(y_pred)
    
    y_t=y_t.cpu()
    y_p=y_p.cpu()
            
    score=roc_auc_score(y_t, y_p)
    
    ns_fpr, ns_tpr, _ = roc_curve(y_t, y_p)
    
    plt.plot(ns_fpr, ns_tpr, marker='.')
    
    running_loss=running_loss/num_batches
    
    plt.plot(ns_fpr, ns_tpr, marker='.')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.savefig("train "+str(1+num_epochs)+"_"+str(threshold)+".jpg")
    
    print(round(score,3))
    print(round(running_loss,3))
    writer.add_scalar('Loss/train', running_loss,(1+num_epochs) * len(dataloader))
    writer.add_scalar('ROC auc score/train', score,(1+num_epochs) * len(dataloader))

    
   
        
def test_model(model, dataloader, loss_fn, device,num_epochs, threshold):

    test_loss=0.0
    
    y_pred = []
    y_true = []
    
    y_pred_restr=[]
    y_true_restr=[]

    num_batches = len(dataloader)

    with torch.no_grad():
        for X, y in dataloader:
            
            X=X.to(device=device)
            y=y.to(device=device)
            
            pred = model(X)
            loss = loss_fn(pred.squeeze(), y)
            
            test_loss += loss.item() # save loss
            
            y = y.data.cpu().numpy()
                    
            y_pred.extend(pred)
            y_true.extend(y) 
            
    for i in range(len(y_true)):
                if y_true[i]==1 or y_true[i]==0:
                    y_pred_restr.append(y_pred[i])
                    y_true_restr.append(y_true[i])
                    
                    
    y_t = torch.FloatTensor(y_true_restr)
    y_p = torch.FloatTensor(y_pred_restr)
    
    y_t=y_t.cpu()
    y_p=y_p.cpu()
            
    score=roc_auc_score(y_t, y_p)
    
    ns_fpr, ns_tpr, thresholds = roc_curve(y_t, y_p)
    
    plt.plot(ns_fpr, ns_tpr, marker='.')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.savefig("test "+str(1+num_epochs)+"_"+str(threshold)+".jpg")
    
    test_loss /= num_batches
    
    writer.add_scalar('Loss/valid', 
                      test_loss, 
                      (1+num_epochs) * len(dataloader))
            
    writer.add_scalar('ROC auc score/valid', 
                      score, 
                      (1+num_epochs) * len(dataloader))
    
    savetxt('data1.csv',ns_fpr,delimiter=',')
    savetxt('data2.csv',ns_tpr,delimiter=',')
    savetxt('data3.csv',thresholds,delimiter=',')
    
    print("Test loss {:.3} Score {:.2}".format(test_loss,score))
    return test_loss,score

if  __name__ == '__main__':
    
    args=get_parser()
    
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate=args.learning_rate
    reg_threshold=args.threshold
    test=args.test_set

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    writer=SummaryWriter()
    model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)
    
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
    
    print("Test set is video"+str(test)+"/"+str(nvideo)+"\n-------------------------------")
    
    # Create training and validation datasets
    
    data_ind=np.arange(video_index[-1]-1)
    test_ind=np.arange(video_index[test-1]-1,video_index[test]-1)
    train_ind=np.delete(data_ind,test_ind)
    
    train_set=torch.utils.data.Subset(dataset,train_ind)
    test_set=torch.utils.data.Subset(dataset,test_ind)
    
    print(len(train_set))
    print(len(test_set))
    print(len(train_set)+len(test_set))
    
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
    torch.save(model_ft.state_dict(), 'model_regression_vid_'+str(test)+'.pth')
    writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, 'threshold': reg_threshold},{'hparam/ROC auc score': sc, 'hparam/loss': tl})
    
    writer.close()