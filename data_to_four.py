# import os
# import shutil
#
# # 定义图片所在的文件夹路径
# image_folder = '沉积岩1'
#
# # 获取所有图片文件
# image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif'))]
#
# # 创建一个字典来存储分类结果
# categories = {}
#
# # 遍历所有图片文件
# for image_file in image_files:
#     # 假设文件名中的汉字部分是分类依据
#     # 这里假设文件名格式为 "汉字_其他信息.jpg"
#     chinese_part = image_file.split('_')[0]  # 根据实际情况调整分割方式
#
#     # 如果该汉字部分还没有对应的分类文件夹，则创建一个
#     if chinese_part not in categories:
#         categories[chinese_part] = []
#         os.makedirs(os.path.join(image_folder, chinese_part), exist_ok=True)
#
#     # 将图片移动到对应的分类文件夹中
#     shutil.move(os.path.join(image_folder, image_file), os.path.join(image_folder, chinese_part, image_file))
#
# print("分类完成！")

import os
from PIL import Image

def split_image(image_path, output_folder):
    # 打开图片
    img = Image.open(image_path)
    width, height = img.size

    # 计算切割后的图片大小
    half_width = width // 2
    half_height = height // 2

    # 切割图片
    img1 = img.crop((0, 0, half_width, half_height))
    img2 = img.crop((half_width, 0, width, half_height))
    img3 = img.crop((0, half_height, half_width, height))
    img4 = img.crop((half_width, half_height, width, height))

    # 获取文件名和扩展名
    filename, ext = os.path.splitext(os.path.basename(image_path))

    # 保存切割后的图片
    img1.save(os.path.join(output_folder, f"{filename}-1{ext}"))
    img2.save(os.path.join(output_folder, f"{filename}-2{ext}"))
    img3.save(os.path.join(output_folder, f"{filename}-3{ext}"))
    img4.save(os.path.join(output_folder, f"{filename}-4{ext}"))

def process_folder(folder_path, output_base_folder):
    # 遍历文件夹中的所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # 获取子文件夹的相对路径
                relative_path = os.path.relpath(root, folder_path)
                output_folder = os.path.join(output_base_folder, relative_path)

                # 创建输出文件夹
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # 切割图片
                split_image(os.path.join(root, file), output_folder)

if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = "沉积岩1"
    # 输出文件夹路径
    output_folder = "沉积岩2"

    # 处理文件夹
    process_folder(input_folder, output_folder)
