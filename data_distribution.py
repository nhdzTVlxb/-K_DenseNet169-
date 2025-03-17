# dataset division into train, test and validation sets; 75%, 15%, 15%
import os
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Path to the directory where the original dataset was uncompressed
org_dataset_dir = "/home/沉积岩/"

# destination directory
base_train = "/home/data_org/train/"
base_test = "/home/data_org/test/"
base_validation = "/home/data_org/validation/"

# create directories
os.makedirs(base_train, exist_ok=True)
os.makedirs(base_test, exist_ok=True)
os.makedirs(base_validation, exist_ok=True)

# random seed
random.seed(42)

# division into 3 sets
train_ratio = 0.70
testF_ratio = 0.30
train_len = 0
test_len = 0
validation_len = 0

categories = [d for d in os.listdir(org_dataset_dir) if os.path.isdir(os.path.join(org_dataset_dir, d))]

for category in tqdm(categories):
    local_dir = os.path.join(org_dataset_dir, category)  # 直接使用类别名称作为路径

    # division into 3 sets
    files = os.listdir(local_dir)
    train, testF = train_test_split(files, train_size=train_ratio, test_size=testF_ratio, shuffle=True)
    test, validation = train_test_split(testF, train_size=0.5, test_size=0.5, shuffle=True)

    # Paths for the new directories
    train_Final = os.path.join(base_train, category)
    test_Final = os.path.join(base_test, category)
    validation_Final = os.path.join(base_validation, category)

    # Create directories for this category
    os.makedirs(train_Final, exist_ok=True)
    os.makedirs(test_Final, exist_ok=True)
    os.makedirs(validation_Final, exist_ok=True)

    # Copy files into the new directories and count them
    for j in train:
        local_file = os.path.join(local_dir, j)  # Path to the file
        shutil.copy(local_file, train_Final)
        train_len += 1

    for j in test:
        local_file = os.path.join(local_dir, j)  # Path to the file
        shutil.copy(local_file, test_Final)
        test_len += 1

    for j in validation:
        local_file = os.path.join(local_dir, j)  # Path to the file
        shutil.copy(local_file, validation_Final)
        validation_len += 1

# Analysis of the sets created
print("总图片数: ", train_len + test_len + validation_len)
print("训练集: ", train_len)
print("测试集: ", test_len)
print("验证集: ", validation_len)
