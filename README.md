# The UNET

## Dataset
### DRIVE 2004
The DRIVE database was established to enable comparative studies on segmentation of blood vessels in retinal images. The dataset is available on [kaggle](https://www.kaggle.com/datasets/zionfuo/drive2004). Sample image and mask are shown below. The dataset contains 20 samples in the training set and 20 samples in the validation set.

![sample](media/sample.png)

## UNET from scratch

With reference to the original UNET paper, I build the UNET architecture from scratch with PyTorch. The architecture is shown below.

I train and evaluate the model on the DRIVE 2004 dataset and visualize its predictions (masks). I experiment with different batch sizes and epochs.  

![architecture](media/architecture.png)
## Using a pretrained encoder

Using the segmentation_models.pytorch [library](https://github.com/qubvel/segmentation_models.pytorch/tree/master), I train a UNET with a pretrained encoder (resnet) on the same DRIVE 2004 dataset.

## Results
The 'ResUNET' was expected to perform much better than the base UNET however this was not the case. The loss curves show that the 'ResUNET' immediately overfits on the training data. This may be due to the size of the training data (only 20 samples) and/or the complexity of the the pretrained model.

### Model without pretrained encoder
![loss_curve](checkpoints/model_without_pretraining/losses.jpg)

### Model with pretrained encoder
![loss_curve](checkpoints/with_pretrained_encoder/losses.jpg)

### Prediciton on test data
![mask](predictions/03_test.gif)

### Ground truth
![mask](archive/DRIVE/test/1st_manual/03_manual1.gif)

Both models learn and generalize well over the training data with ResUNET overfitting. The model's prediction is decent but can be significantly improved. There is a lot more than can be done to make this work much better.  

## TODO
- [x] Experiment pretrained encoder
- [x] Use command line arguments
- [ ] Augment data and retrain model
- [ ] Experiment with other UNET varients (pretrained encoders)
- [ ] Publish medium article to explain work further

## Additional 
Code was deveoped locally whilst training runs were done in a colab[notebook](https://colab.research.google.com/drive/1ryi9zquGhvP_p5DFmuPyO1mrG1d7fRFz?usp=sharing) and on aws notebook instance (ml.g4dn.xlarge, 1 T4 GPU)

## File Structure

```├── archive/ # Dataset
│ ├── DRIVE/
│ │ ├── training/ # Train data
│ │ └── test/ # Test data
├── checkpoints/ # Loss curves and weights from experiments
├── media/ # Images, papers
├── predictions/ # Output from inferencing
├── training.py # Custom script for training models
├── .gitignore # List of files and folders ignored by git
├── infer.py # Custom script for running inference
├── dataset.py # Custom dataset class
├── utils.py # Helper functions
└── unet.py # Model architectures
```

