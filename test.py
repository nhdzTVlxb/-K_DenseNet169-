import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision import models, datasets, transforms
from torchvision.models import DenseNet169_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# 定义一个类，用于同时将输出写入文件和终端
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()

# 打开文件用于保存输出
log_file = "/home/Rock/K_Fold_cross_validation/DenseNet/log5.txt"
file = open(log_file, "w")
sys.stdout = Tee(sys.stdout, file)  # 将标准输出重定向到文件和终端

# 数据路径
d = "/home/Rock/data_org/"
fld = 'PyDL_C'
test_dir = d + "test/"
validation_dir = d + "validation/"  # 验证集数据路径

# 定义图像转换器
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 将 PIL 图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载测试集数据
original_test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# 加载验证集数据
original_validation_dataset = datasets.ImageFolder(root=validation_dir, transform=transform)

# 获取类别名称
class_names = original_test_dataset.classes

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试集 DataLoader
batch_size = 16
test_loader = DataLoader(original_test_dataset, batch_size=batch_size, shuffle=False)

# 验证集 DataLoader
val_loader = DataLoader(original_validation_dataset, batch_size=batch_size, shuffle=False)

# 类别数量
num_classes = len(class_names)

# 测试函数
def test_model(model, test_loader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return y_true, y_pred

# 计算特异性（Specificity）
def calculate_specificity(y_true, y_pred, class_index):
    tn = 0
    fp = 0
    for true, pred in zip(y_true, y_pred):
        if true != class_index and pred != class_index:
            tn += 1
        elif true != class_index and pred == class_index:
            fp += 1
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity

# 计算多类平均特异性
def calculate_multi_class_specificity(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    specificity_scores = []
    for class_index in range(num_classes):
        specificity = calculate_specificity(y_true, y_pred, class_index)
        specificity_scores.append(specificity)
    return np.mean(specificity_scores)

# 加载模型并测试
acc = []
val_acc = []
f1_scores = []
precisions = []
recalls = []
specificities = []
y_true_ll = []  # 保存每个 fold 的真实标签
y_pred_ll = []  # 保存每个 fold 的预测标签

for f in range(5):
    # 加载模型
    model = models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    
    # 加载训练好的权重
    path = d + fld + '/models/DenseNet169_fold_T_' + str(f+1) +'.pth'
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))  # 使用 map_location 确保加载到正确设备
    model.to(device)
    
    # 测试模型
    y_true, y_pred = test_model(model, test_loader)
    y_true_ll.append(y_true)  # 保存真实标签
    y_pred_ll.append(y_pred)  # 保存预测标签
    
    # 计算测试集准确率
    acc.append(accuracy_score(y_true, y_pred))
    print(f"Fold {f+1}: Test Accuracy: {acc[-1] * 100}%")

    # 计算验证集准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:  # 使用验证集 DataLoader
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    val_acc.append(val_accuracy)
    print(f"Fold {f+1}: Validation Accuracy: {val_accuracy}%")

    # 计算其他指标
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    specificity = calculate_multi_class_specificity(y_true, y_pred)
    
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)
    specificities.append(specificity)
    
    # 输出每个 fold 的详细指标
    print(f"Fold {f+1}: F1 Score: {f1}")
    print(f"Fold {f+1}: Precision: {precision}")
    print(f"Fold {f+1}: Recall/Sensitivity: {recall}")
    print(f"Fold {f+1}: Specificity: {specificity}")

# 输出平均指标
print(f"\nAverage Test Accuracy: {np.mean(acc) * 100}%")
print(f"Average Validation Accuracy: {np.mean(val_acc)}%")
print(f"Average F1 Score: {np.mean(f1_scores)}")
print(f"Average Precision: {np.mean(precisions)}")
print(f"Average Recall/Sensitivity: {np.mean(recalls)}")
print(f"Average Specificity: {np.mean(specificities)}")

# 分类报告
for i in range(5):
    print(f"\nFold {i+1} Classification Report:")
    print(classification_report(y_true_ll[i], y_pred_ll[i], target_names=class_names))

# 手动加载字体
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 使用 Regular 样式
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 生成混淆矩阵
for i in range(5):
    cm = confusion_matrix(y_true_ll[i], y_pred_ll[i])
    
    # 设置图片大小和字体
    plt.figure(figsize=(50, 50))  # 调整图片大小
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        cmap=plt.cm.Blues,  # 颜色映射
        xticks_rotation='vertical',  # 旋转 x 轴标签
        values_format='d',  # 显示整数
        text_kw={'fontsize': 2},  # 设置字体大小
        include_values=True,  # 显示数值
        colorbar=True  # 添加颜色条
    )
    plt.title(f"Fold {i+1} Confusion Matrix")
    plt.tight_layout()  # 自动调整布局，避免标题或标签被截断
    plt.savefig(f'/home/Rock/line4/confusion_matrix_fold_{i+1}.png')
    plt.close()

# 关闭文件
file.close()
