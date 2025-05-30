import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 配置路径和参数
query_dir = "5_query_images"  # 检索请求图像路径
db_dir = "45_database_images"  # 被检索图像路径
output_dir = "results"  # 结果保存路径
os.makedirs(output_dir, exist_ok=True)


# 计算RGB直方图（8 bins/通道）
def compute_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB格式
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 归一化并展平为1D向量
    return hist


# 加载所有图像直方图
database = {}
for img_name in os.listdir(db_dir):
    img_path = os.path.join(db_dir, img_name)
    database[img_name] = compute_histogram(img_path)

# 处理每个检索请求
for query_name in os.listdir(query_dir):
    query_path = os.path.join(query_dir, query_name)
    query_hist = compute_histogram(query_path)

    # 计算与所有被检索图像的相似度（欧氏距离）
    distances = {}
    for db_name, db_hist in database.items():
        distance = np.linalg.norm(query_hist - db_hist)
        distances[db_name] = distance

    # 按距离排序，取前3名
    sorted_results = sorted(distances.items(), key=lambda x: x[1])[:3]

    # 可视化结果
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 4, 1)
    query_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')

    # 显示前3结果
    for i, (result_name, distance) in enumerate(sorted_results):
        result_path = os.path.join(db_dir, result_name)
        result_img = cv2.cvtColor(cv2.imread(result_path), cv2.COLOR_BGR2RGB)
        plt.subplot(1, 4, i + 2)
        plt.imshow(result_img)
        plt.title(f"Rank {i + 1}\nDist: {distance:.2f}")
        plt.axis('off')

    plt.suptitle(f"Query: {query_name}", fontsize=12)
    plt.savefig(os.path.join(output_dir, f"result_{query_name}.png"))
    plt.close()

print("检索完成！结果已保存至", output_dir)