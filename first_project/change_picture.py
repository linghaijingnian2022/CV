import cv2
import numpy as np

# 读取彩色图像
I0 = cv2.imread("I0.jpeg")

# ------------------------------
# 1. 亮度变换（+30亮度值）
# ------------------------------
brightness = 30  # 亮度调整值
I_bright = cv2.addWeighted(I0, 1, np.zeros_like(I0), 0, brightness)

# ------------------------------
# 2. 对比度变换（alpha=1.5, beta=30）
# ------------------------------
alpha = 1.5  # 对比度增强系数
beta = 30     # 亮度补偿
I_contrast = cv2.convertScaleAbs(I0, alpha=alpha, beta=beta)

# ------------------------------
# 3. 饱和度变换（HSV空间S通道×1.5）
# ------------------------------
hsv = cv2.cvtColor(I0, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)  # 饱和度增强1.5倍
I_saturation = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# 保存结果
cv2.imwrite("I_bright.jpg", I_bright)
cv2.imwrite("I_contrast.jpg", I_contrast)
cv2.imwrite("I_saturation.jpg", I_saturation)