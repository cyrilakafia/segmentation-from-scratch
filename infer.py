import torch
from unet import UNet
from utils import load_checkpoint
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import io


CHECKPOINT = 'checkpoints/checkpoint_10.pth.tar'
IMG_PATH = '/home/cyril/unet_segmentation_from_scratch/archive/DRIVE/training/images/21_training.tif'

def main():
    model = UNet()

    load_checkpoint(torch.load(CHECKPOINT), model)
    
    with torch.no_grad():
        image = np.array(Image.open(IMG_PATH).convert("RGB"))
        image = torch.rand((2, 3, 572, 572)) 

        output = model(image)
        
        output = output[0][0]
        
        output = np.array(output)
    
        # plt.imsave('mask.gif', output)
        
        image = Image.fromarray(output)
        image.save('predictions/mask.gif')
        
        io.imsave('predictins/mask_1.gif', output)
image = np.array(Image.open('/home/cyril/unet_segmentation_from_scratch/archive/DRIVE/training/1st_manual/21_manual1.gif').convert("L"))
print(image.shape)

if __name__ == "__main__":
    main()
    




