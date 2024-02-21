import csv
import random


# 生成假姓名的函数
def generate_fake_name():
    first_names = ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Henry", "Ivy", "Jack"]
    last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"


# 生成CSV文件
csv_file_path = "example_data/sample_data.csv"
field_names = ["Name", "Age", "City"]

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=field_names)

    # 写入CSV文件的表头
    writer.writeheader()

    # 生成并写入200条数据
    for _ in range(200):
        name = generate_fake_name()
        age = random.randint(18, 60)
        city = random.choice(["New York", "London", "Tokyo", "Paris", "Berlin"])

        writer.writerow({"Name": name, "Age": age, "City": city})

print(f"CSV文件已生成：{csv_file_path}")
