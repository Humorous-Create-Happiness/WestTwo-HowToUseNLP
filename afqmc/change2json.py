input_file = "dev.json"  # 输入文件路径
output_file = "dev_processed.json"  # 输出文件路径

# 读取输入文件并转换格式
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 在每一行的末尾添加逗号
lines = [line.strip() + "," + "\n" for line in lines]

# 保存为新的文件
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("文件转换完成！")

