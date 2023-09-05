import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNet, UNet_Pretrained
from utils import get_loaders, save_checkpoint, load_checkpoint, plot_metrics
import torch.nn.functional as F
import os

import numpy as np
from PIL import Image
import argparse


parser = argparse.ArgumentParser(prog='Training and Validation', description='Training and Validation a segmentation model')
parser.add_argument('epochs', type=int, help='Number of epochs')
parser.add_argument('batch', type=int, help='Batch size')
parser.add_argument('experiment', type=str, help='Name of traning experiment')
parser.add_argument('pretrained', type=bool, help='Use the pretrained model')

args = parser.parse_args()

# hyperparameters 
EXPERIMENT_NAME = args.experiment
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.batch
NUM_EPOCHS = args.epochs
NUM_WORKERS = 2
PRETRAINED = args.pretrained
if PRETRAINED:
    IMAGE_HEIGHT = 576
    IMAGE_WIDTH = 576
else:
    IMAGE_HEIGHT = 572
    IMAGE_WIDTH = 572
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "archive/DRIVE/training/images/"
TRAIM_MASK_DIR = "archive/DRIVE/training/1st_manual"
VAL_IMG_DIR = "archive/DRIVE/test/images"
VAL_MASK_DIR = "archive/DRIVE/test/1st_manual"


def train(train_loader, val_loader, model, optimizer, criterion, epochs):
    train_loss_history = []
    val_loss_history = []
    
    print(f'Training starting: learning_rate={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={NUM_EPOCHS}, pretrained_encoder={PRETRAINED}')
    
    for epoch in range(epochs):
        
        train_loss = 0
        val_loss = 0
        
        model.train()
        
        loop = tqdm(train_loader, ncols=80, desc='Training')
    
        for i, (input, target) in enumerate(loop):
            input = input.to(device = DEVICE)
            target = target.float().unsqueeze(1).to(device = DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(input)
            
            target = F.interpolate(target, size=outputs.size()[2:], mode='nearest')
            
            loss = criterion(outputs, target)
            
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())
        
        model.eval()
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, ncols=80, desc='Validation')
            
            for i, (input, target) in enumerate(val_loop):
                input = input.to(device = DEVICE)
                target = target.float().unsqueeze(1).to(device=DEVICE)
                optimizer.zero_grad()
                ouputs = model(input)
                
                target = F.interpolate(target, size=outputs.size()[2:], mode='nearest')
                
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                
                val_loop.set_postfix(val_loss=loss.item())
    
        
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if not os.path.exists(f'checkpoints/{EXPERIMENT_NAME}'):
            os.makedirs(f'checkpoints/{EXPERIMENT_NAME}')
            
        save_checkpoint(checkpoint, filename=f'checkpoints/{EXPERIMENT_NAME}/weights_{NUM_EPOCHS}_epoch.pth.tar')
        
        epoch_loss = train_loss/len(train_loader)
        val_epoch_loss = val_loss/len(val_loader)
        
        print(f'Epoch {epoch + 1} | Training loss : {epoch_loss:.5f} | Validation loss : {val_epoch_loss}')
        
        train_loss_history.append(epoch_loss)
        val_loss_history.append(val_epoch_loss)
        
    save_checkpoint(model, filename=f'checkpoints/{EXPERIMENT_NAME}/last_{NUM_EPOCHS}_epoch.pth')
        
    return model, train_loss_history, val_loss_history, optimizer, criterion

def main():
    if PRETRAINED:
        model = UNet_Pretrained().to(device=DEVICE)
        
    else:
        model = UNet().to(device=DEVICE)
        
    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], 
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    
    test_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    train_loader, val_loader = get_loaders(
        train_dir=TRAIN_IMG_DIR, 
        train_maskdir=TRAIM_MASK_DIR, 
        val_dir=VAL_IMG_DIR, 
        val_maskdir=VAL_MASK_DIR, 
        train_transform=train_transforms, 
        val_transform=test_transforms,
        batch_size=BATCH_SIZE
        )
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    _, train_loss_history, val_loss_history, _, _ = train(train_loader, val_loader, model, optimizer, criterion, epochs=NUM_EPOCHS)

    plot_metrics(train_loss_history, val_loss_history, NUM_EPOCHS, experiment_name = EXPERIMENT_NAME)
if __name__ == "__main__":
    main()

