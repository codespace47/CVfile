import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


# 1. 核心修复：解决matplotlib显示中文报错
def fix_matplotlib_chinese():
    """修复matplotlib无法显示中文的问题，兼容Windows/Linux/Mac"""
    # 设置字体（优先系统自带中文字体）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows：黑体；Linux/Mac可替换为 'WenQuanYi Micro Hei'/'PingFang SC'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    # 验证字体是否生效（可选）
    try:
        from matplotlib.font_manager import FontManager
        fm = FontManager()
        mat_fonts = set(f.name for f in fm.ttflist)
        chinese_font = 'SimHei'
        if chinese_font in mat_fonts:
            print(f"中文显示字体 {chinese_font} 加载成功")
        else:
            print(f"未找到{chinese_font}，将使用默认兼容字体")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    except:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("系统字体检测失败，使用默认字体（中文可能显示为方框，建议手动安装中文字体）")


# 执行中文显示修复
fix_matplotlib_chinese()


# 2. 定义模型结构
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# 3. 加载训练好的模型
def load_trained_model(model_path):
    """加载已保存的模型权重"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = MNIST_CNN().to(device)

    # 加载权重（兼容CPU/GPU训练的模型）
    try:
        # 解决GPU训练模型在CPU上加载的问题
        if device.type == 'cpu':
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_path))
        model.eval()  # 切换到评估模式（禁用Dropout）
        print(f"模型加载成功：{model_path}")
        return model, device
    except Exception as e:
        raise ValueError(f"模型加载失败：{e}\n请检查模型文件路径是否正确，或模型结构是否与训练时一致")


# 4. 复用图片分割和预处理函数
def split_student_id_image(image_path):
    """分割单张10位学号图片，返回数字张量和原始裁剪图"""
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}，请检查路径是否正确")

    # 预处理：灰度化→二值化→去噪声
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 轮廓检测
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选数字轮廓
    digit_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) >= 50 and 0.2 <= w / h <= 2:
            digit_contours.append((x, y, w, h, cnt))

    # 按x坐标排序
    digit_contours.sort(key=lambda x: x[0])
    if len(digit_contours) != 10:
        print(f"检测到{len(digit_contours)}个数字轮廓（预期10个），请检查照片质量")
        # 可视化轮廓
        img_contour = cv2.drawContours(img.copy(), [cnt for _, _, _, _, cnt in digit_contours], -1, (0, 255, 0), 2)
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
        plt.title(f'检测到的轮廓数量：{len(digit_contours)}（预期10个）', fontsize=12)
        plt.axis('off')
        plt.show()

    # 预处理每个数字为模型输入格式
    digit_tensors = []
    digit_origin_imgs = []
    for x, y, w, h, cnt in digit_contours[:10]:
        # 裁剪并扩边
        x_start = max(0, x - 2)
        y_start = max(0, y - 2)
        x_end = min(binary.shape[1], x + w + 2)
        y_end = min(binary.shape[0], y + h + 2)
        digit_roi = binary[y_start:y_end, x_start:x_end]

        # 调整尺寸+归一化
        digit_roi = cv2.resize(digit_roi, (28, 28))
        digit_roi = digit_roi / 255.0
        digit_roi = (digit_roi - 0.1307) / 0.3081

        # 转为张量
        digit_tensor = torch.from_numpy(digit_roi).float().unsqueeze(0).unsqueeze(0)
        digit_tensors.append(digit_tensor)
        digit_origin_imgs.append(digit_roi)

    return digit_tensors, digit_origin_imgs, img


def test_student_id_recognition(model_path, image_path):
    """
    完整的测试流程：加载模型 → 分割图片 → 识别 → 可视化（中文标题）
    """
    # 1. 加载模型
    model, device = load_trained_model(model_path)

    # 2. 分割图片
    digit_tensors, digit_imgs, origin_img = split_student_id_image(image_path)

    # 3. 识别数字
    model.eval()
    student_id = ""
    with torch.no_grad():
        for tensor in digit_tensors:
            tensor = tensor.to(device)
            outputs = model(tensor)
            _, predicted = torch.max(outputs.data, 1)
            student_id += str(predicted.item())

    # 4. 可视化结果（含中文）
    plt.figure(figsize=(18, 8))

    # 子图1：原始学号照片
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB))
    plt.title(f'原始学号照片', fontsize=14)
    plt.axis('off')

    # 子图2：分割后的数字+识别结果（中文标题）
    plt.subplot(2, 1, 2)
    n = len(digit_imgs)
    for i, (img, pred) in enumerate(zip(digit_imgs, student_id)):
        plt.subplot(2, n, n + i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'数字{i + 1}：识别结果 {pred}', fontsize=10)
        plt.axis('off')

    plt.suptitle(f'学号识别结果：{student_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 输出最终结果
    print(f"\n最终识别的10位学号：{student_id}")
    return student_id


# -------------------------- 6. 执行测试（修改路径！） --------------------------
if __name__ == "__main__":
    # ========== 请修改为你的实际路径 ==========
    MODEL_PATH = "./mnist_model.pth"  # 训练好的模型文件路径
    IMAGE_PATH = "C://Users//hp//Desktop//student_id.jpg"  # 你的10位学号照片路径

    # 执行测试
    try:
        result = test_student_id_recognition(MODEL_PATH, IMAGE_PATH)
    except Exception as e:
        print(f"测试过程出错：{e}")