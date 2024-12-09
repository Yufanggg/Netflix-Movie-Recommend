# 数据文件路径
from matplotlib import pyplot as plt

file_path = "output.txt"

# 读取数据并解析
data = []
with open(file_path, "r") as f:
    for line in f:
        # 将字符串转换为元组
        row = eval(line.strip())
        data.append(row)

# 过滤第三列大于 0.5 的数据
filtered_data = [row for row in data if row[2] > 0.5]

# 按第三列从大到小排序
sorted_filtered_data = sorted(filtered_data, key=lambda x: x[2], reverse=True)

sorted_data = sorted(sorted_filtered_data, key=lambda x: x[2])  # 按第三列从低到高排序
third_column_sorted = [item[2] for item in sorted_data]

# 可视化排序后的第三列数据
plt.figure(figsize=(12, 6))
plt.plot(range(len(third_column_sorted)), third_column_sorted, marker='o', linestyle='-', color='g', label='Similarity Score (Sorted)')
plt.title('Visualization of Similarity Scores', fontsize=16)
plt.xlabel('Index (Sorted)', fontsize=12)
plt.ylabel('Similarity Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

print(len(sorted_filtered_data))