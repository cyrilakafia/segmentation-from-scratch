import torch
from unet import UNet
from utils import load_checkpoint, load_pretrained_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = 'checkpoints/checkpoint_10.pth.tar'
IMG_PATH = 'archive/DRIVE/training/images/21_training.tif'
IMG_NAME = IMG_PATH.split('/')[-1][:-3]
PRETRAINED = False
if PRETRAINED:
    IMAGE_HEIGHT = 576
    IMAGE_WIDTH = 576
else:
    IMAGE_HEIGHT = 572
    IMAGE_WIDTH = 572
    

def main():
    
    if PRETRAINED:
        model = load_pretrained_model(CHECKPOINT)
        
    else:
        model = UNet()

        load_checkpoint(torch.load(CHECKPOINT), model)
    
    model.to(device=DEVICE)
    model.eval()
    
    with torch.no_grad():
        image = np.array(Image.open(IMG_PATH).convert("RGB"))
        
        transform = A.Compose([  
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), 
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],  max_pixel_value=255.0), 
            ToTensorV2()]
        ) 
        
        transformed_image = transform(image=image)
        image = transformed_image["image"]
        image = image.unsqueeze(0)
        image = image.to(device=DEVICE)
        
        output = torch.sigmoid(model(image))
        
        output = output[0][0].cpu().numpy() 
        
        output *= 255.0
        
        output = output.astype(np.uint8)
        
        image = Image.fromarray(output)
        
        if not os.path.exists('predictions'):
            os.mkdir('predictions')
        
        image.save(f'predictions/{IMG_NAME}gif')

if __name__ == "__main__":
    main()
    




