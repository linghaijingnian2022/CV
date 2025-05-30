import cv2

# 读取彩色图像（替换为你的图片路径）
I0 = cv2.imread('I0.jpeg')

# 转为灰度图像
I1 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)

# 保存结果
cv2.imwrite('I1.jpg', I1)