import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torchvision import datasets
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import models
from torchvision.models import DenseNet169_Weights

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import cv2

from tqdm import tqdm
from colorama import Fore, Style

# colorama
red = Fore.RED
green = Fore.GREEN
blue = Fore.BLUE
yellow = Fore.YELLOW
cyan = Fore.CYAN

reset = Style.RESET_ALL

# Data
d = "/home/Rock/data_org/"
fld = 'PyDL_C'

# Sub-Categorized data
train_dir = d + "train/"
test_dir = d + "test/"
validation_dir = d + "validation/"

# Setting the seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

print(f'{blue}Global seed set to : {yellow}{seed}\n')

img_dimen = (256, 256)
bs = 16

# preprocessing | get the data mean and std for normalization
transform = transforms.Compose([
    transforms.Resize(img_dimen),
    transforms.ToTensor()
])

calc_ms = datasets.ImageFolder(root=train_dir, transform=transform)
loader_ms = torch.utils.data.DataLoader(dataset=calc_ms, batch_size=bs, shuffle=False)

mean_calc = 0
std_calc = 0
total_images = 0

for images, _ in tqdm(loader_ms):
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean_calc += images.mean(2).sum(0)
    std_calc += images.std(2).sum(0)
    total_images += batch_samples

mean_calc /= total_images
std_calc /= total_images

print(f'{blue}mean: {yellow}{mean_calc}')
print(f'{blue}std: {yellow}{std_calc}{reset}')

# ImageNet Normalization
# mean_calc = [0.485, 0.456, 0.406]
# std_calc = [0.229, 0.224, 0.225]

# 自定义 Dataset 类以支持 albumentations
class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)  # 将 PIL 图像转换为 NumPy 数组
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

# Data transformations training set
transform_all = A.Compose([
    A.RandomResizedCrop(height=img_dimen[0], width=img_dimen[1], p=0.7),  # 随机裁剪，减少为 0.7
    A.RandomRotate90(p=0.5),  # 随机旋转 90°
    A.HorizontalFlip(p=0.5),  # 随机水平翻转
    A.Transpose(p=0.5),  # 随机转置
    A.OneOf([
        A.MotionBlur(p=0.1),  # 运动模糊
        A.MedianBlur(blur_limit=3, p=0.1),  # 中值模糊
        A.Blur(blur_limit=3, p=0.1),  # 模糊
    ], p=0.1),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=30, p=0.2),  # 平移、缩放、旋转
    A.OneOf([
        A.OpticalDistortion(p=0.1),  # 光学畸变，降低到 0.1
        A.GridDistortion(p=0.1),  # 网格畸变
    ], p=0.1),
    A.OneOf([
        A.CLAHE(clip_limit=2, p=0.2),  # 对比度受限自适应直方图均衡化
        A.RandomBrightnessContrast(p=0.2),  # 随机亮度对比度调整
    ], p=0.2),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),  # 色调、饱和度、明度调整
    A.Resize(height=img_dimen[0], width=img_dimen[1], p=1.0),  # 强制调整所有图像大小
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ToTensorV2()  # 转换为 PyTorch 张量
])

# 验证和测试集的增强方法
transform_test = A.Compose([
    A.Resize(height=img_dimen[0], width=img_dimen[1], p=1.0),  # 调整大小
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ToTensorV2()  # 转换为 PyTorch 张量
])

# 加载原始数据集（ImageFolder）
original_train_dataset = datasets.ImageFolder(root=train_dir)
original_test_dataset = datasets.ImageFolder(root=test_dir)

# 获取类别名称
class_names = original_train_dataset.classes

# 将原始数据集包装为 AlbumentationsDataset
dataset_train = AlbumentationsDataset(original_train_dataset, transform=transform_all)
dataset_test = AlbumentationsDataset(original_test_dataset, transform=transform_test)

# 创建 DataLoader
batch_size = bs
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# 类别数量
num_classes = len(class_names)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{blue}Device: {yellow}{device}{reset}')

# Hyperparameters
fold = 5
max_epoch = 50
batch_size = 16
learningRate = 0.0001
WeightDecay = 1e-08

# All Information
print(f'{blue}Fold: {yellow}{fold}{reset}')
print(f'{blue}Epochs: {yellow}{max_epoch}{reset}')
print(f'{blue}Batch size: {yellow}{batch_size}{reset}')
print(f'{blue}Learning rate: {yellow}{learningRate}{reset}')
print(f'{blue}Weight decay: {yellow}{WeightDecay}{reset}')

# k-fold cross-validation
kf = KFold(n_splits=fold, shuffle=True, random_state=seed)

# K fold cross-validation

# Define your train and validation scores for all folds
# Loss metrics
train_loss_all = []
val_loss_all = []
# Accuracy metrics
train_acc_all = []
val_acc_all = []

# validation accuracy for calculating average
fold_val_acc = []

# Loop over each fold
for fold, (train_index, val_index) in enumerate(kf.split(dataset_train)):    
    print(f'{yellow}\n##############################################')
    print(f'{green}                   FOLD {fold + 1}')
    print(f'{yellow}##############################################{reset}')

    # Define your train and validation datasets
    train_dataset = Subset(dataset_train, train_index)
    val_dataset = Subset(dataset_train, val_index)

    # Define your train and validation dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # -----------------------------------------------------------------------------------
    
    # DenseNet169
    model = models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)

    num_classes = len(dataset_train.dataset.classes)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes)

    model.to(device)
    
    # -----------------------------------------------------------------------------------
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learningRate, weight_decay=WeightDecay)
    
    # -----------------------------------------------------------------------------------  
    
    # TRAINING
    # loss metrics
    train_loss = []
    val_loss = []
    # Accuracy metrics
    train_acc = []
    val_acc = []

    # Max score for the current fold
    max_curr_fold = 0

    # Loop over each epoch
    for epoch in range(max_epoch):
        model.train()

        # Metrics initialization
        running_loss = 0.0
        num_correct = 0

        # TRAINING
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Predictions | forward pass | OUTPUT
            outputs = model(inputs)
            # Loss | backward pass | GRADIENT
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            # Count correct predictions
            num_correct += (predicted == labels).sum().item()
            
        # ---------------------------------------------------------------------------
        # Training loss
        train_lss = running_loss / len(train_loader)
        train_loss.append(train_lss)

        # Training accuracy
        train_accuracy = 100 * num_correct / len(train_loader.dataset)
        train_acc.append(train_accuracy)
        # ---------------------------------------------------------------------------

        model.eval()
        correct = 0
        valid_loss = 0

        # VALIDATION
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # Predictions
                outputs = model(inputs)
                # Count correct predictions
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                # Loss
                valid_loss += criterion(outputs, labels).item()

        # --------------------------------------------------------------------------
        #Validation loss
        val_lss = valid_loss / len(val_loader)
        val_loss.append(val_lss)

        # Validation accuracy
        val_accuracy = 100 * correct / len(val_loader.dataset)
        val_acc.append(val_accuracy)
        
        # --------------------------------------------------------------------------
        
        print(f'{cyan}\nEPOCH {epoch + 1}{reset}')
        print(f"Loss: {red}{train_lss}{reset}, Validation Accuracy: {red}{val_accuracy}%{reset}, Training Accuracy: {red}{train_accuracy}%")
        
        # Save the best model of each fold
        if val_accuracy > max_curr_fold:
            max_curr_fold = val_accuracy
            ff = fold + 1
            path = d + fld + '/models/DenseNet169_fold_T_' + str(ff) +'.pth'
            torch.save(model.state_dict(), path)
            print(f'{green}Improvement! Model saved!{reset}')
            
    # save last model
    ff = fold + 1
    path = d + fld + '/models/DenseNet169_fold_F_' + str(ff) +'.pth'
    torch.save(model.state_dict(), path)
    
    # ------------------------------------------------------------------------------
    
    # metrics for graph for current fold
    train_loss_all.append(train_loss)
    val_loss_all.append(val_loss)
    
    train_acc_all.append(train_acc)
    val_acc_all.append(val_acc)
    
    # the highest validation accuracy of each fold       
    fold_val_acc.append(max_curr_fold)
    
    # ------------------------------------------------------------------------------
        
print(f'{yellow}\nTraining finished!')

# Graph of training and validation: loss and accuracy | dual plots for each fold
fig, axis = plt.subplots(5, 2, figsize=(20, 40))

for i in range(5):
    # Loss plot
    axis[i, 0].set_title("Fold " + str(i+1) + ": Loss")
    axis[i, 0].plot(val_loss_all[i], color='red', label='Validation loss', linestyle='dashed')
    axis[i, 0].plot(train_loss_all[i], color='orange', label='Training loss')
    axis[i, 0].legend()
    axis[i, 0].set_xlabel("Iterations")
    axis[i, 0].set_ylabel("Loss")

    # Accuracy plot
    axis[i, 1].set_title("Fold " + str(i+1) + ": Accuracy")
    axis[i, 1].plot(val_acc_all[i], color='red', label='Validation accuracy', linestyle='dashed')
    axis[i, 1].plot(train_acc_all[i], color='orange', label='Training accuracy')
    axis[i, 1].legend()
    axis[i, 1].set_xlabel("Iterations")
    axis[i, 1].set_ylabel("Accuracy")

plt.savefig('/home/Rock/line/training_validation_curves.png')
plt.close()  # 关闭图形，释放内存

import numpy as np

# Graph of training and validation: loss and accuracy | single plot for all folds
fig, axis = plt.subplots(1, 2, figsize=(20, 10))

acc_mean = []
loss_mean = []

for i in range(5):
    acc_mean.append(sum(val_acc_all[i]) / len(val_acc_all[i]))
    loss_mean.append(sum(val_loss_all[i]) / len(val_loss_all[i]))
    
acc_std = []
loss_std = []

for i in range(5):
    acc_std.append(np.std(val_acc_all[i]))
    loss_std.append(np.std(val_loss_all[i]))
    
# Loss plot
axis[0].set_title("Loss")
axis[0].errorbar(range(1, 6), loss_mean, yerr=loss_std, color='red', label='Validation loss', linestyle='dashed')
axis[0].plot(range(1, 6), loss_mean, color='orange', label='Training loss')
axis[0].legend()
axis[0].set_xlabel("Folds")
axis[0].set_ylabel("Loss")

# Accuracy plot
axis[1].set_title("Accuracy")
axis[1].errorbar(range(1, 6), acc_mean, yerr=acc_std, color='red', label='Validation accuracy', linestyle='dashed')
axis[1].plot(range(1, 6), acc_mean, color='orange', label='Training accuracy')
axis[1].legend()
axis[1].set_xlabel("Folds")
axis[1].set_ylabel("Accuracy")

plt.savefig('/home/Rock/line/average_training_validation_curves.png')
plt.close()  # 关闭图形，释放内存
