import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNet, UNet_Pretrained
from utils import get_loaders, save_checkpoint, load_checkpoint
import torch.nn.functional as F
import os

# hyperparameters 
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 250
NUM_WORKERS = 2
PRETRAINED = True
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

def train(loader, model, optimizer, criterion, epochs):
    train_loss_history = []
    train_accuracy_history = []
    
    print(f'Training starting: learning_rate={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={NUM_EPOCHS}, pretrained_encode{PRETRAINED}')
    
    for epoch in range(epochs):
        
        train_loss = 0
        
        model.train()
        
        loop = tqdm(loader, ncols=80)
    
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
        
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
            
        save_checkpoint(checkpoint, filename=f'checkpoints/checkpoint_{NUM_EPOCHS}.pth.tar')
        
        epoch_loss = train_loss/len(loader)
        
        print(f'Epoch {epoch + 1} | Training loss : {epoch_loss:.5f}')
        
        train_loss_history.append(epoch_loss)
        
    save_checkpoint(model, filename=f'checkpoints/last_{NUM_EPOCHS}.pth')
        
    return model, train_loss_history, optimizer, criterion

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
    
    train_loader, test_loader = get_loaders(
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
    
    sample_img, annot = next(iter(train_loader))
    # print(sample_img.shape)
    
    train(train_loader, model, optimizer, criterion, epochs=NUM_EPOCHS)
    

if __name__ == "__main__":
    main()

