# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:19:59 2022

@author: Utente
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score,confusion_matrix

from create_dataset_noblack import FrameDataset

#let's define some parameters

root_dir='frames'
csv_file='annotations.csv'


num_classes = 3

n_frames=238078
n_frames_0 = 72583
n_frames_1= 54797
n_frames_2=110698

classes=['stable','arising','unstable']

weights=torch.tensor([1/n_frames_0,1/n_frames_1,1/n_frames_2])



# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of epochs. Default value 100.", default=5)
    parser.add_argument("--learning-rate", type=float, help="Learning rate. Default value 0.001.", default=0.001)
    parser.add_argument("--batch-size", type=int, help="Batch size. Default value 10.", default=10)
    parser.add_argument("--load", type=bool, help="Load. Default value False.", default=False)
    args = parser.parse_args()
    return args

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            

def initialize_model(num_classes, feature_extract,load, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
   
    model_ft = models.resnet101(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    if load:
        model_ft.load_state_dict(torch.load('model_full.pth'))
    
    return model_ft, input_size

def train_model(model, dataloader, loss_fn, optimizer, writer, device, num_epochs, is_inception=False):
    
    i=0
    
    running_loss=0.0
    accuracy=0
    
    y_pred = []
    y_true = []
    
    
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device=device)
        y = y.to(device=device)
        
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #data storing
        
        running_loss += loss.item() # save loss
        
        output = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        y = y.data.cpu().numpy()
        y_true.extend(y) # Save truth
        
        accuracy=balanced_accuracy_score(y_true,y_pred)
        
        i+=1
        if batch % 100 == 0:
            
            steps = num_epochs * len(dataloader) + i # calculate steps 
            batch = i*batch_size # calculate batch 
            print("Training loss {:.3} Accuracy {:.3} Steps: {}".format(running_loss / batch, accuracy, steps))
            
            # Save accuracy and loss to Tensorboard
            writer.add_scalar('Loss/train', running_loss / batch, steps)
            writer.add_scalar('Accuracy/train', accuracy, steps)
            
        cf_matrix = confusion_matrix(y_true, y_pred,normalize="true")
    
        df_cm = pd.DataFrame(cf_matrix , index=[i for i in classes],
                     columns=[i for i in classes])
        plt.figure(figsize=(12, 7))  
        writer.add_figure("Confusion matrix/train", sn.heatmap(df_cm, annot=True).get_figure(), num_epochs)
        
        
    
def test_model(model, dataloader, loss_fn, device,num_epochs):

    test_loss=0.0
    accuracy = 0
    
    y_pred = []
    y_true = []

    num_batches = len(dataloader)

    with torch.no_grad():
        for X, y in dataloader:
            
            X=X.to(device=device)
            y=y.to(device=device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            
            test_loss += loss.item() # save loss
            
            output = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
        
            y = y.data.cpu().numpy()
            y_true.extend(y) # Save truth
            
            accuracy=balanced_accuracy_score(y_true,y_pred)
            
    cf_matrix = confusion_matrix(y_true, y_pred,normalize="true")
    
    df_cm = pd.DataFrame(cf_matrix , index=[i for i in classes],
                     columns=[i for i in classes])
    plt.figure(figsize=(12, 7))  
    writer.add_figure("Confusion matrix/valid", sn.heatmap(df_cm, annot=True).get_figure(), num_epochs)
    
    test_loss /= num_batches
    
    writer.add_scalar('Loss/valid', 
                      test_loss, 
                      (1+num_epochs) * len(dataloader))
            
    writer.add_scalar('Accuracy/valid', 
                      accuracy, 
                      (1+num_epochs) * len(dataloader))
    
    print(f"Test Error: \n Accuracy: {(accuracy)*100:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss,accuracy

if  __name__ == '__main__':
    
    args=get_parser()
    
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate=args.learning_rate
    load=args.load

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    writer=SummaryWriter()
    model_ft, input_size = initialize_model(num_classes, feature_extract,load, use_pretrained=True)
    
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
    criterion = nn.CrossEntropyLoss(weight=weights)
    
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
        train_model(model_ft, train_loader, criterion, optimizer_ft, writer, device, t)
        tl,ta= test_model(model_ft, test_loader, criterion, device,t )
    print("Done!")
    
    model_ft=model_ft.to(device='cpu')
    torch.save(model_ft.state_dict(), 'model_full.pth')
    writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},{'hparam/accuracy': ta, 'hparam/loss': tl})
    
    writer.close()
    
    #testare separando i video in video di train e video di test
    #protocollo sperimentale alternando tutti i vari video come test (leave one out) V
    
    #i valori oltre i limiti vengono clippati agli estremi inizialmente
    #provare poi con sigmoide, probabilmente peggio
    
    #confrontare con accuratezza bilanciata del metodo esperto
    
    #una via è sfruttare dei threshold e costruire delle classi
    #altrimenti culva ROC,prende y predette e vere (binarie) e identifica una curva sulla base di ciò
    #sklearn ROC out score, o curva plottabile, in questo caso si esclude i valori intermedi
    
    
    