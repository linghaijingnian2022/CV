import cv2
import numpy as np

# 读取I1（灰度图）和I2（二值图）
I1 = cv2.imread("I1_gray.jpeg", cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread("I2_binary.jpg", cv2.IMREAD_GRAYSCALE)

# 确保I2与I1分辨率相同，并转为二值掩码（0或1）
I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))
_, I2_bin = cv2.threshold(I2, 128, 1, cv2.THRESH_BINARY)  # 二值化为0/1

# 为每个位平面生成替换后的图像
for bit in range(1, 9):
    bit_pos = bit - 1  # 位平面位置（从0开始计数）
    # 构建清零当前位的掩码，避免负数产生
    mask = np.uint8(255 - (1 << bit_pos))
    cleared = I1 & mask     # 清除原图的当前位
    new_bits = I2_bin << bit_pos  # 将二值图左移到目标位
    replaced = cleared | new_bits # 合并生成新图像
    cv2.imwrite(f"L{bit}_replaced.jpg", replaced)