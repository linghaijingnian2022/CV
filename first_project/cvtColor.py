import cv2

# 读取彩色图像（BGR格式）
I0 = cv2.imread("I0.jpeg")  # 请确保图片路径正确

# 转换为灰度图像
I1 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)

# 保存结果
cv2.imwrite("I1.jpeg", I1)