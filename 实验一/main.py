import cv2
import numpy as np
import matplotlib.pyplot as plt


# 手动灰度化 Y = 0.299R + 0.587G + 0.114B
def rgb_to_gray(img_rgb):
    gray = (
        0.299 * img_rgb[:, :, 0] +
        0.587 * img_rgb[:, :, 1] +
        0.114 * img_rgb[:, :, 2]
    )
    return gray.astype(np.float32)


# 手写二维卷积
def conv2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


# 手写颜色直方图
def color_histogram(channel):
    hist = np.zeros(256, dtype=np.int32)
    h, w = channel.shape
    for i in range(h):
        for j in range(w):
            hist[channel[i, j]] += 1
    return hist


def main():
    # 读取图像
    img = cv2.imread('C://Users//hp//Desktop//firstMotor.png')
    if img is None:
        print("图像读取失败，请检查路径")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 灰度图
    gray = rgb_to_gray(img_rgb)

    # Sobel 算子
    sobel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [1,  2,  1],
        [0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float32)

    gx = conv2d(gray, sobel_x)
    gy = conv2d(gray, sobel_y)
    sobel = np.sqrt(gx ** 2 + gy ** 2)

    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    laplacian_result = conv2d(gray, laplacian_kernel)
    laplacian_result = np.abs(laplacian_result)
    laplacian_result = (laplacian_result / laplacian_result.max()) * 255

    # 颜色直方图
    r_hist = color_histogram(img_rgb[:, :, 0])
    g_hist = color_histogram(img_rgb[:, :, 1])
    b_hist = color_histogram(img_rgb[:, :, 2])

    # 纹理特征
    mean_gradient = np.mean(sobel)
    std_gradient = np.std(sobel)
    max_gradient = np.max(sobel)

    texture_feature = np.array([
        mean_gradient,
        std_gradient,
        max_gradient
    ])

    np.save("texture_feature.npy", texture_feature)

    print("纹理特征：", texture_feature)

    # 可视化
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Gray Image")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(sobel, cmap='gray')
    plt.title("Sobel Result")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(laplacian_result, cmap='gray')
    plt.title("Laplacian Edge")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.plot(r_hist, color='r', label='R')
    plt.plot(g_hist, color='g', label='G')
    plt.plot(b_hist, color='b', label='B')
    plt.title("Color Histogram")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
