import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    添加高斯噪声
    - mean: 噪声均值
    - sigma: 噪声标准差（控制噪声强度）
    """
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(image, prob=0.05):
    """
    添加椒盐噪声
    - prob: 噪声点出现的总概率（盐噪声和胡椒噪声各占一半）
    """
    output = np.copy(image)
    # 生成随机矩阵标记噪声位置
    half_prob = prob / 2
    random_matrix = np.random.random(image.shape)
    # 椒噪声（黑色，像素值=0）
    output[random_matrix < half_prob] = 0
    # 盐噪声（白色，像素值=255）
    output[random_matrix > 1 - half_prob] = 255
    return output

def add_uniform_noise(image, intensity=50):
    """
    添加均匀分布噪声
    - intensity: 噪声强度（噪声范围是 [-intensity/2, intensity/2]）
    """
    uniform_noise = np.random.uniform(-intensity/2, intensity/2, image.shape)
    noisy = image.astype(np.float32) + uniform_noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# 读取灰度图像（假设已生成 I1）
I1 = cv2.imread('I1.jpg', cv2.IMREAD_GRAYSCALE)

# 生成带噪声的图像
I_gaussian = add_gaussian_noise(I1, sigma=30)  # 高斯噪声
I_salt_pepper = add_salt_pepper_noise(I1, prob=0.1)  # 椒盐噪声
I_uniform = add_uniform_noise(I1, intensity=80)  # 均匀噪声

# 保存结果
cv2.imwrite('noisy_gaussian.jpg', I_gaussian)
cv2.imwrite('noisy_salt_pepper.jpg', I_salt_pepper)
cv2.imwrite('noisy_uniform.jpg', I_uniform)

# 均值滤波器
def mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

# 中值滤波器
def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# 高斯滤波器
def gaussian_filter(image, kernel_size=3, sigma=0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

# 去噪处理
gaussian_mean = mean_filter(I_gaussian)
gaussian_median = median_filter(I_gaussian)
gaussian_gaussian = gaussian_filter(I_gaussian)

salt_pepper_mean = mean_filter(I_salt_pepper)
salt_pepper_median = median_filter(I_salt_pepper)
salt_pepper_gaussian = gaussian_filter(I_salt_pepper)

uniform_mean = mean_filter(I_uniform)
uniform_median = median_filter(I_uniform)
uniform_gaussian = gaussian_filter(I_uniform)

# 保存结果
cv2.imwrite('gaussian_mean.jpg', gaussian_mean)
cv2.imwrite('gaussian_median.jpg', gaussian_median)
cv2.imwrite('gaussian_gaussian.jpg', gaussian_gaussian)

cv2.imwrite('salt_pepper_mean.jpg', salt_pepper_mean)
cv2.imwrite('salt_pepper_median.jpg', salt_pepper_median)
cv2.imwrite('salt_pepper_gaussian.jpg', salt_pepper_gaussian)

cv2.imwrite('uniform_mean.jpg', uniform_mean)
cv2.imwrite('uniform_median.jpg', uniform_median)
cv2.imwrite('uniform_gaussian.jpg', uniform_gaussian)