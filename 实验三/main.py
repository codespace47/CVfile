import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. 配置环境和超参数
device = torch.device("cpu")
print(f"使用设备: {device}")

batch_size = 64
learning_rate = 0.001
epochs = 5  # 新手够用，想更准可调至10

# 2. 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#  3. 定义CNN模型（复用，识别单数字）
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


# 初始化模型、损失函数、优化器
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 4. 训练+测试模型
def train_model():
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0


def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'模型测试准确率: {accuracy:.2f}%')
    return accuracy


# 执行训练和测试
print("开始训练模型...")
train_model()
test_model()


# -------------------------- 5. 核心：单张学号图片分割函数（重点新增） --------------------------
def split_student_id_image(image_path):
    """
    分割单张包含10位学号的图片，返回按顺序排列的10个数字的预处理张量
    步骤：灰度化→二值化→去噪声→找轮廓→筛选数字轮廓→排序→裁剪+预处理
    """
    # 1. 读取图片并转为灰度图
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}，请检查路径是否正确")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 二值化（增强对比度，数字为黑，背景为白）
    # 自适应二值化，适合不同光照的照片
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 形态学操作：去除小噪声（如墨点）
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 4. 查找所有轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 筛选数字轮廓（按面积、宽高比，排除过小/过大的噪声）
    digit_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤条件：面积≥50（排除小噪声），宽高比0.2~2（排除长条/扁条噪声）
        if cv2.contourArea(cnt) >= 50 and 0.2 <= w / h <= 2:
            digit_contours.append((x, y, w, h, cnt))

    # 6. 按x坐标排序（保证从左到右的数字顺序）
    digit_contours.sort(key=lambda x: x[0])

    # 检查是否分割出10个数字
    if len(digit_contours) != 10:
        print(f"警告：检测到{len(digit_contours)}个数字轮廓（预期10个），请检查照片是否清晰/数字间距适中")
        # 若分割数量不对，可视化轮廓帮助排查
        img_contour = cv2.drawContours(img.copy(), [cnt for _, _, _, _, cnt in digit_contours], -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
        plt.title(f'检测到{len(digit_contours)}个轮廓')
        plt.axis('off')
        plt.show()

    # 7. 对每个数字轮廓裁剪+预处理（转为MNIST格式）
    digit_tensors = []
    digit_origin_imgs = []  # 保存原始裁剪图，用于可视化
    for x, y, w, h, cnt in digit_contours[:10]:  # 只取前10个
        # 裁剪数字区域（上下左右各扩2像素，避免切到数字）
        x_start = max(0, x - 2)
        y_start = max(0, y - 2)
        x_end = min(binary.shape[1], x + w + 2)
        y_end = min(binary.shape[0], y + h + 2)
        digit_roi = binary[y_start:y_end, x_start:x_end]

        # 调整为28x28（MNIST尺寸）
        digit_roi = cv2.resize(digit_roi, (28, 28))

        # 归一化（和MNIST训练数据一致）
        digit_roi = digit_roi / 255.0
        digit_roi = (digit_roi - 0.1307) / 0.3081

        # 转为张量并添加维度（[1,1,28,28]，适配模型输入）
        digit_tensor = torch.from_numpy(digit_roi).float().unsqueeze(0).unsqueeze(0)
        digit_tensors.append(digit_tensor.to(device))

        # 保存原始裁剪图（用于可视化）
        digit_origin_imgs.append(digit_roi)

    return digit_tensors, digit_origin_imgs


# -------------------------- 6. 识别分割后的数字 --------------------------
def recognize_student_id(image_path):
    """
    输入单张学号图片路径，返回识别出的10位学号
    """
    model.eval()
    # 分割图片得到数字张量
    digit_tensors, digit_imgs = split_student_id_image(image_path)

    # 逐个识别数字
    student_id = ""
    with torch.no_grad():
        for tensor in digit_tensors:
            outputs = model(tensor)
            _, predicted = torch.max(outputs.data, 1)
            student_id += str(predicted.item())

    # 可视化分割和识别结果
    plt.figure(figsize=(15, 3))
    for i, (img, pred) in enumerate(zip(digit_imgs, student_id)):
        plt.subplot(1, len(digit_imgs), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'识别: {pred}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    return student_id


# -------------------------- 7. 实际使用（修改图片路径！） --------------------------
# ========== 关键：修改为你的10位学号照片路径 ==========
student_id_image_path = "C://Users//hp//Desktop//student_id.jpg"  # 你的单张学号照片路径（如jpg/png）

# 执行识别
try:
    result = recognize_student_id(student_id_image_path)
    print(f"\n最终识别出的10位学号: {result}")

    # 保存模型（可选）
    torch.save(model.state_dict(), "./mnist_model.pth")
    print("模型已保存为 mnist_model.pth")

except Exception as e:
    print(f"识别过程出错: {e}")