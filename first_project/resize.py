import cv2
import numpy as np

# 读取原灰度图像I1获取分辨率
I1 = cv2.imread("I1.jpeg", cv2.IMREAD_GRAYSCALE)
target_height, target_width = I1.shape[:2]  # 获得目标分辨率

# 读取手写照片
I2 = cv2.imread("handwritten.jpg")

# 调整分辨率（强制匹配I1尺寸）
resized = cv2.resize(I2, (target_width, target_height), interpolation=cv2.INTER_AREA)

# 转为灰度
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# 二值化（Otsu自适应阈值）
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 保存结果
cv2.imwrite("I2_binary.jpg", binary)