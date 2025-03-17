import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F
from collections import defaultdict

# 定义设备：使用GPU或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 类别标签映射字典
class_names = ["中性岩", "云英岩", "凝灰岩", "千枚岩", "变粒岩", "含变余构造变质岩", "基性岩", "大理岩", 
               "斜长角闪岩", "板岩", "榴辉岩", "沉凝灰岩", "泥页岩", "混合岩", "灰岩", "片岩", "片麻岩", "白云岩", 
               "矽卡岩", "砂岩", "硅质岩", "硬石膏", "碎裂岩", "碱性岩及相关岩石", "磷块岩", "粉砂岩", "糜棱岩", 
               "绿片岩", "蛇纹岩", "赤铁矿岩", "超基性（超镁铁质）岩", "酸性岩", "铝土矿", "麻粒岩"]

# 定义岩石大类
sedimentary_rocks = ["凝灰岩", "沉凝灰岩", "泥页岩", "灰岩", "白云岩", "砂岩", "硅质岩", "硬石膏", "磷块岩", "粉砂岩", "赤铁矿岩", "铝土矿"]
igneous_rocks = ["中性岩", "酸性岩", "碱性岩及相关岩石", "基性岩", "超基性（超镁铁质）岩"]
metamorphic_rocks = [class_name for class_name in class_names if class_name not in sedimentary_rocks and class_name not in igneous_rocks]

# 加载模型
def load_model(weight_path):
    model = models.densenet169(pretrained=False)  # 使用DenseNet169架构
    model.classifier = torch.nn.Linear(model.classifier.in_features, 34)  # 修改输出类别数为34
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 定义图片预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # 增加批次维度
    return img.to(device)

# 模型集成预测函数
def ensemble_predict(models, image_path):
    img = preprocess_image(image_path)
    
    # 存储所有模型的预测结果
    outputs = []
    
    with torch.no_grad():
        for model in models:
            output = model(img)
            outputs.append(output)
    
    # 计算模型集成的平均预测结果
    avg_output = torch.mean(torch.stack(outputs), dim=0)
    
    # 获取最终预测的类别（取最大值的索引）
    _, predicted_class = torch.max(avg_output, 1)
    
    return predicted_class.item()

# 预测并保存结果到文件
def predict_and_save(models, image_folder, output_file):
    # 用来统计每类岩石的数量
    class_counts = defaultdict(int)
    class_images = defaultdict(list)  # 存储每个类的图片列表
    sedimentary_counts = 0
    igneous_counts = 0
    metamorphic_counts = 0
    sedimentary_images = []
    igneous_images = []
    metamorphic_images = []

    with open(output_file, 'w') as f:
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                predicted_class = ensemble_predict(models, image_path)
                
                # 获取对应类别名称
                predicted_class_name = class_names[predicted_class]
                
                # 更新每个类别的统计信息
                class_counts[predicted_class_name] += 1
                class_images[predicted_class_name].append(image_name)
                
                # 判断所属岩石大类
                if predicted_class_name in sedimentary_rocks:
                    sedimentary_counts += 1
                    sedimentary_images.append(image_name)
                elif predicted_class_name in igneous_rocks:
                    igneous_counts += 1
                    igneous_images.append(image_name)
                elif predicted_class_name in metamorphic_rocks:
                    metamorphic_counts += 1
                    metamorphic_images.append(image_name)

                # 打印结果到终端
                print(f"Image: {image_name}, Predicted Class: {predicted_class_name}")
                
                # 将结果写入文件
                f.write(f"Image: {image_name}, Predicted Class: {predicted_class_name}\n")
    
    # 输出每类岩石的数量
    print("\n每类岩石的数量：")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} 张图片")
    
    # 输出每个岩石大类的数量及其图片
    print("\n每类岩石大类的数量：")
    print(f"沉积岩: {sedimentary_counts} 张图片")
    for image in sedimentary_images:
        print(f"  {image}")
    
    print(f"火成岩: {igneous_counts} 张图片")
    for image in igneous_images:
        print(f"  {image}")
    
    print(f"变质岩: {metamorphic_counts} 张图片")
    for image in metamorphic_images:
        print(f"  {image}")
    
    # 输出每个岩石类对应的图片
    print("\n每个岩石类对应的图片：")
    for class_name, images in class_images.items():
        print(f"{class_name}:")
        for image in images:
            print(f"  {image}")

# 主程序
if __name__ == '__main__':
    # 模型权重文件列表（现在有五个权重文件）
    weight_files = [
        '/home/Rock/data_org/PyDL_C/models/DenseNet169_fold_T_1.pth',
        '/home/Rock/data_org/PyDL_C/models/DenseNet169_fold_T_2.pth',
        '/home/Rock/data_org/PyDL_C/models/DenseNet169_fold_T_3.pth',
        '/home/Rock/data_org/PyDL_C/models/DenseNet169_fold_T_4.pth',
        '/home/Rock/data_org/PyDL_C/models/DenseNet169_fold_T_5.pth'
    ]

    # 加载所有模型
    models = [load_model(weight_path) for weight_path in weight_files]

    # 待预测图片文件夹
    image_folder = '/home/Rock/附件/'
    
    # 输出结果的文件路径
    output_file = '/home/Rock/line/ansower.txt'
    
    # 进行预测并保存结果
    predict_and_save(models, image_folder, output_file)
