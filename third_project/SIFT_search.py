import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 配置路径
query_dir = "5_query_images"  # 检索请求图像路径
db_dir = "45_database_images"  # 被检索图像路径
output_dir = "sift_results"  # 结果保存路径
os.makedirs(output_dir, exist_ok=True)

# 初始化SIFT检测器和匹配器
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # 或使用FLANN匹配器

# 预提取数据库图像的SIFT特征
database_features = {}
for img_name in os.listdir(db_dir):
    img_path = os.path.join(db_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    if des is not None:
        database_features[img_name] = des  # 存储描述子

# 处理每个查询图像
for query_name in os.listdir(query_dir):
    query_path = os.path.join(query_dir, query_name)
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    kp_query, des_query = sift.detectAndCompute(query_img, None)

    if des_query is None:
        print(f"查询图像 {query_name} 未检测到SIFT特征，跳过。")
        continue

    # 存储匹配结果（格式：{图像名: 匹配点数量}）
    matches_count = {}

    # 与所有数据库图像匹配
    for db_name, db_des in database_features.items():
        if db_des is None:
            continue
        # 使用k-NN匹配（k=2）并应用比率测试
        matches = bf.knnMatch(des_query, db_des, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # 比率测试阈值
                good_matches.append(m)
        matches_count[db_name] = len(good_matches)  # 记录匹配点数量

    # 按匹配点数量排序，取前3名
    sorted_results = sorted(matches_count.items(), key=lambda x: -x[1])[:3]

    # 可视化结果
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB))
    plt.title("Query Image")
    plt.axis('off')

    # 显示前3结果
    for i, (result_name, count) in enumerate(sorted_results):
        result_path = os.path.join(db_dir, result_name)
        result_img = cv2.cvtColor(cv2.imread(result_path), cv2.COLOR_BGR2RGB)
        plt.subplot(1, 4, i + 2)
        plt.imshow(result_img)
        plt.title(f"Rank {i + 1}\nMatches: {count}")
        plt.axis('off')

    plt.suptitle(f"Query: {query_name}", fontsize=12)
    plt.savefig(os.path.join(output_dir, f"result_{query_name}.png"))
    plt.close()

print("SIFT检索完成！结果已保存至", output_dir)