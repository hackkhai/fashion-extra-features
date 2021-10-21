import os

#from __future__ import print_function
#from __future__ import division

# Pytorch DL framework imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler,DataLoader

# Standard python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import os
import copy
from PIL import *

# Progress bar
from tqdm import tqdm

# Function for shuffling the dataset
from sklearn.utils import shuffle

# Imports for calculating metrics
import ml_metrics as metrics

# contemporary CNN architecture
from efficientnet_pytorch import EfficientNet

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
balance=True
interrupted=False

# Batch size for training (change depending on how much memory you have)
batch_size = 50

# Number of epochs to train for
num_epochs = 14
workers = 0
# Number of classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); device
#paths 
CKP_DIR="./training"
DATA_DIR= './'
IMG_DIR_TRAIN = './train/'
IMG_DIR_TEST = './train/'
data_csv= os.path.join(DATA_DIR,'features_w_names_reduced.csv')
chunk=pd.read_csv(data_csv,chunksize=1000000)
df = pd.concat(chunk)
# df = df.iloc[:100,:]
num_classes = len(df.columns) - 1
class CustomDatasetFromCSV(Dataset):
    def __init__(self, df, transformations, folder):
        """
        Args:
            csv_path (string): path to csv file
            transformations: pytorch transforms for transforms and tensor conversion
            train: flag to determine if train or val set
        """
        # Transforms
        self.transforms = transformations
        # Second column is the photos
        self.image_arr = np.asarray(df.iloc[:, 0])
        
        # Second column is the labels
        self.label_arr = np.asarray(df.iloc[:, 1:-1])
        
        # Calculate len
        self.data_len = len(self.label_arr)
        #Init path to folder with photos
        self.folder=folder

    def __getitem__(self, index):

        # Get image name from the pandas Series
        single_image_name = self.image_arr[index]
        # Open image and convert to RGB (some dataset images are grayscale)
        img_as_img = Image.open(os.path.join(self.folder, single_image_name+".jpg")).convert('RGB')

        #Use transforms
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)

        #Get image labels from the pandas DataFrame
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
#load pretrined efficientnet (https://arxiv.org/abs/1905.11946) 
model_ft= EfficientNet.from_pretrained("efficientnet-b3", num_classes=num_classes)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Linear(num_ftrs,len(cats))
# )
#get defolt input size 
#get defolt input size 
input_size = EfficientNet.get_image_size('efficientnet-b3')
# Img model input size
im_size = input_size

# df=df.iloc[:,1:]
#add augmentations
#add augmentations
transform_train = transforms.Compose([transforms.Resize((im_size, im_size)),                           
                                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomAffine(5),
                                transforms.RandomRotation(180),
                               transforms.ToTensor(),
                               #transforms.Lambda(lambda img: img * 2.0 - 1.0)
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
# Transformations on inference
transform_val = transforms.Compose([transforms.Resize((im_size, im_size)),
                               transforms.ToTensor(),
                               #transforms.Lambda(lambda img: img * 2.0 - 1.0)
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])

# df=pd.read_csv(data_csv)
# df.head()

df["weight"]=1
train=df.sample(frac=0.9, random_state=42)
# print(train.iloc[:,2:-1].head())
val=df.drop(train.index)
#calculate weight of classes to improve balance 
print("here")
if balance==True:
    weight=train.iloc[:,1:].sum()
    def make_weight(x, weight):
        return min(weight[x[1:]==1])
    train.weight=train.apply(make_weight, args=(weight,), axis=1)
    train.weight=sum(weight)/train.weight


trainset = CustomDatasetFromCSV(train, transform_train, IMG_DIR_TRAIN)
valset = CustomDatasetFromCSV(val, transform_val, IMG_DIR_TRAIN)


if balance==True:
    class_weights_train=torch.tensor(train.weight.values)

    weighted_sampler_train = WeightedRandomSampler(
        weights=class_weights_train,
        num_samples=len(class_weights_train),
        replacement=True
    )

    dataloaders_dict = {"train":DataLoader(trainset , shuffle=False , batch_size=batch_size, sampler=weighted_sampler_train, num_workers=workers),
                  "val": DataLoader(valset , shuffle=False , batch_size=batch_size, num_workers=workers)}
else:
    dataloaders_dict = {"train":DataLoader(trainset , shuffle=True , batch_size=batch_size, num_workers=workers),
                  "val": DataLoader(valset , shuffle=False , batch_size=batch_size, num_workers=workers)}
# Function for calculating MEAN Average Precision(MAP) score
def calc_map(preds, labels):
    preds = np.around(preds.cpu().detach().numpy())
    labels = labels.cpu().detach().numpy()             
    pred = []
    for i in preds:
        cats = np.nonzero(list(i))[0]
        pred.append(list(cats))
    label = []
    for i in labels:
        cats = np.nonzero(list(i))[0]
        label.append(list(cats))
    return metrics.mapk(label, pred)
def train_model(model, dataloaders, criterion, optimizer,path_to_ckp, scheduler, num_epochs=25, star_epoch=0):
    since = time.time()

    val_map_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mapk = 0.0
    for epoch in range(star_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_mapk = 0.0
            # Iterate over data.
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    labels = labels.float()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        # For multi-label
                        outputs = torch.sigmoid(outputs)
                        loss = criterion(outputs, labels)

                        preds = outputs

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # statistics
                    ma = calc_map(preds, labels.data)
                    running_loss += loss.item() * inputs.size(0)
                    running_mapk += ma
                    tepoch.set_postfix(loss = loss.item() * inputs.size(0),mAP = ma)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_mapk = running_mapk / len(dataloaders[phase])

            #scheduler step
            if phase=='train':
                scheduler.step()

            print('{} Loss: {:.4f}  MAP: {:.4f}'.format(phase, epoch_loss,  epoch_mapk))


            if phase == 'val' and epoch_mapk > best_mapk:
                best_mapk = epoch_mapk
                best_model_wts = copy.deepcopy(model.state_dict())
                #save checkpoint
                chp={
                    "model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "epoch":epoch,
                    "scheduler":scheduler.state_dict()
                }
                torch.save(chp,os.path.join(path_to_ckp, "model_best"+".pt"))
            if epoch%1==0:
                best_model_wts = copy.deepcopy(model.state_dict())
                #save checkpoint
                chp={
                    "model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "epoch":epoch,
                    "scheduler":scheduler.state_dict()
                }
                torch.save(chp,os.path.join(path_to_ckp, "model_"+str(epoch)+".pt"))
            if phase == 'val':
                val_map_history.append(epoch_mapk)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val MAP: {:4f}'.format(best_mapk))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_map_history
#unfreeze percent of layers
def unfreeze(model,percent=0.25):
    l = int(np.ceil(len(model._modules.keys())* percent))
    l = list(model._modules.keys())[-l:]
    print(f"unfreezing these layer {l}",)
    for name in l:
        for params in model._modules[name].parameters():
            params.requires_grad_(True)
#freeze all layers
#freeze all layers
for param in model_ft.parameters():
    param.requires_grad = False
#unfreeze 60% of layers
unfreeze(model_ft, 0.6)
#unfreeze 60% of convolutional bloks
unfreeze(model_ft._blocks, 0.6)
def check_freeze(model):
    for name ,layer in model._modules.items():
        s = []
        for l in layer.parameters():
            s.append(l.requires_grad)
        print(name ,all(s))
check_freeze(model_ft)
model_ft = model_ft.to(device)
#https://arxiv.org/pdf/2009.14119.pdf
#I have no idea why it work worse then classic BCELoss at this case
class AsymmetricLossOptimized(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
def focal_binary_cross_entropy(logits, targets, gamma=2):
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where(t >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = num_label*loss.mean()
    return loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        self._gamma = gamma
        self._alpha = alpha

    def forward(self, y_true, y_pred):
        cross_entropy_loss = torch.nn.BCELoss(y_true, y_pred)
        p_t = ((y_true * y_pred) +
               ((1 - y_true) * (1 - y_pred)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (y_true * self._alpha +
                                   (1 - y_true) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    cross_entropy_loss)
        return focal_cross_entropy_loss.mean()
optimizer_ft=torch.optim.Adam(model_ft.parameters(),lr=0.0001)
# criterion=AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05,disable_torch_grad_focal_loss=True)
criterion = torch.nn.BCELoss()
#Scheduler for linear learning rate reduction
from torch.optim import lr_scheduler
scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma = 0.1)
epoch=0

interrupted = False
#load a checkpoint if the training session was interrupted
if interrupted:
    ckp=torch.load(CKP_DIR+"/model_best-Copy1.pt")
    model_ft.load_state_dict(ckp["model"])
    epoch=ckp["epoch"]+1
    optimizer_ft.load_state_dict(ckp["optimizer"])
    scheduler.load_state_dict(ckp["scheduler"])
    print("checkpoint loaded")
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, CKP_DIR, scheduler, num_epochs=num_epochs, star_epoch=epoch)
PATH="./model_final.pt"
torch.save(model_ft.state_dict(), PATH)
