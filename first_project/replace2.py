import cv2
import numpy as np

# 读取原始彩色图像I0和二值图像I2
I0 = cv2.imread("I0.jpeg")  # OpenCV读取为BGR格式
I2 = cv2.imread("I2_binary.jpg", cv2.IMREAD_GRAYSCALE)
I2 = cv2.resize(I2, (I0.shape[1], I0.shape[0]))  # 确保尺寸一致
_, I2_bin = cv2.threshold(I2, 128, 1, cv2.THRESH_BINARY)  # 二值化为0/1

# 分解BGR通道
B, G, R = cv2.split(I0.astype(np.uint8))

def replace_channel_bit(channel, bit_plane, replace_mask):
    """替换指定通道的某一位平面"""
    mask = np.uint8(255 - (1 << (bit_plane-1)))  # 清零掩码
    cleared = channel & mask
    new_bits = replace_mask << (bit_plane-1)
    return cleared | new_bits

# 示例：替换R通道的L8位 + G通道的L1位
modified_R = replace_channel_bit(R, 8, I2_bin)  # R通道最高位替换
modified_G = replace_channel_bit(G, 1, I2_bin)  # G通道最低位替换
modified_B = B.copy()  # 保持B通道不变

# 合并通道并保存结果
merged = cv2.merge([modified_B, modified_G, modified_R])
cv2.imwrite("R8_G1_replaced.jpg", merged)

modified_R1 = R.copy()
modified_G1 = replace_channel_bit(G, 8, I2_bin)
modified_B1 = replace_channel_bit(B, 8, I2_bin)

merged = cv2.merge([modified_B1, modified_G1, modified_R1])
cv2.imwrite("G8_B8_replaced.jpg", merged)

modified_R2 = replace_channel_bit(R, 1, I2_bin)
modified_G2 = replace_channel_bit(G, 1, I2_bin)
modified_B2 = replace_channel_bit(B, 1, I2_bin)
merged = cv2.merge([modified_B2, modified_G2, modified_R2])
cv2.imwrite("B1_G1_R1_replaced.jpg", merged)
