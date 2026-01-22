import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    return edges


def region_of_interest(edges):
    height, width = edges.shape
    # 梯形顶点：仅覆盖道路区域（根据当前图的双向车道调整）
    polygons = np.array([
        [(width//4, height),          # 左下：往中间收，避开路边草地
         (width//2 - 20, int(height*0.3)),  # 左上：更窄，贴合道路消失点
         (width//2 + 20, int(height*0.3)),  # 右上：对称窄区域
         (width*3//4, height)]        # 右下：往中间收
    ], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygons, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges


def detect_lane_lines(image, masked_edges):
    height, width = image.shape[:2]
    # 收紧霍夫参数：只检测长、连续的强车道线段
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,          # 更高：需要更多投票才判定为直线
        minLineLength=80,      # 更长：过滤短杂线
        maxLineGap=30          # 更小：只拼接紧密的车道线段
    )

    # 收窄斜率范围，仅保留车道线的合理斜率
    lane_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            k = (y2 - y1) / (x2 - x1)
            # 收窄斜率：仅保留双向车道的典型斜率
            if -0.7 < k < -0.4 or 0.4 < k < 0.7:
                lane_lines.append([x1, y1, x2, y2])

    return lane_lines


def draw_lane_lines(image, lane_lines):
    line_image = np.zeros_like(image)
    for line in lane_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return result


def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("错误")
        return
    # 读取图像 → 预处理（灰度+模糊+Canny边缘检测）→ 提取道路ROI（排除路边干扰）→ 霍夫变换检测直线 → 斜率过滤筛选车道线 → 绘制车道线 → 保存/展示结果
    edges = preprocess_image(image)
    masked_edges = region_of_interest(edges)
    lane_lines = detect_lane_lines(image, masked_edges)
    result = draw_lane_lines(image, lane_lines)

    # 结果展示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("jpg")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result_rgb)
    plt.title("result")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("lane_detection_result.jpg", dpi=300, bbox_inches="tight")
    plt.show()

    cv2.imwrite("lane_detection_output.jpg", result)
    print("检测完成！结果已保存为 lane_detection_output.jpg")


if __name__ == "__main__":
    main("road.jpg")