import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    添加高斯噪声
    - mean: 噪声均值
    - sigma: 噪声标准差（控制噪声强度）
    """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(image, prob=0.05):
    """
    添加椒盐噪声
    - prob: 噪声点出现的总概率（盐噪声和胡椒噪声各占一半）
    """
    output = np.copy(image)
    half_prob = prob / 2
    random_matrix = np.random.random(image.shape[:2])
    for c in range(image.shape[2]):
        output[random_matrix < half_prob, c] = 0
        output[random_matrix > 1 - half_prob, c] = 255
    return output

def add_uniform_noise(image, intensity=50):
    """
    添加均匀分布噪声
    - intensity: 噪声强度（噪声范围是 [-intensity/2, intensity/2]）
    """
    noise = np.random.uniform(-intensity / 2, intensity / 2, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# 均值滤波器
def mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

# 中值滤波器
def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# 高斯滤波器
def gaussian_filter(image, kernel_size=3, sigma=0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

# 读取彩色图像
I0 = cv2.imread('I0.jpeg')
if I0 is None:
    print("无法读取图像，请检查文件路径。")
else:
    # 生成带噪声的图像
    I0_gaussian = add_gaussian_noise(I0, sigma=30)  # 高斯噪声
    I0_salt_pepper = add_salt_pepper_noise(I0, prob=0.1)  # 椒盐噪声
    I0_uniform = add_uniform_noise(I0, intensity=80)  # 均匀噪声

    # 去噪处理
    gaussian_mean = mean_filter(I0_gaussian)
    gaussian_median = median_filter(I0_gaussian)
    gaussian_gaussian = gaussian_filter(I0_gaussian)

    salt_pepper_mean = mean_filter(I0_salt_pepper)
    salt_pepper_median = median_filter(I0_salt_pepper)
    salt_pepper_gaussian = gaussian_filter(I0_salt_pepper)

    uniform_mean = mean_filter(I0_uniform)
    uniform_median = median_filter(I0_uniform)
    uniform_gaussian = gaussian_filter(I0_uniform)

    # 保存结果
    cv2.imwrite('I0_noisy_gaussian.jpg', I0_gaussian)
    cv2.imwrite('I0_noisy_salt_pepper.jpg', I0_salt_pepper)
    cv2.imwrite('I0_noisy_uniform.jpg', I0_uniform)

    cv2.imwrite('I0_gaussian_mean.jpg', gaussian_mean)
    cv2.imwrite('I0_gaussian_median.jpg', gaussian_median)
    cv2.imwrite('I0_gaussian_gaussian.jpg', gaussian_gaussian)

    cv2.imwrite('I0_salt_pepper_mean.jpg', salt_pepper_mean)
    cv2.imwrite('I0_salt_pepper_median.jpg', salt_pepper_median)
    cv2.imwrite('I0_salt_pepper_gaussian.jpg', salt_pepper_gaussian)

    cv2.imwrite('I0_uniform_mean.jpg', uniform_mean)
    cv2.imwrite('I0_uniform_median.jpg', uniform_median)
    cv2.imwrite('I0_uniform_gaussian.jpg', uniform_gaussian)