import re
import matplotlib.pyplot as plt

# 读取日志文件
log_file = "log.txt"
with open(log_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 用正则表达式提取每行的 step 和各个损失
pattern = re.compile(
    r"Step (\d+).*?Loss: ([\d\.]+).*?Loss1: ([\d\.]+).*?Loss2: ([\d\.]+).*?Del Loss: ([\d\.]+)"
)

steps, total_loss, loss1, loss2, del_loss = [], [], [], [], []

for line in lines:
    match = pattern.search(line)
    if match:
        steps.append(int(match.group(1)))
        total_loss.append(float(match.group(2)))
        loss1.append(float(match.group(3)))
        loss2.append(float(match.group(4)))
        del_loss.append(float(match.group(5)))

if not steps:
    print("❌ 未能从日志中解析出任何数据，请检查日志格式。")
    exit()

print(f"✅ 解析出 {len(steps)} 条记录。")

# 绘图
plt.figure(figsize=(12, 8))

# 总损失
plt.subplot(2, 2, 1)
plt.plot(steps, total_loss, label="Total Loss", marker="o")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Total Loss")
plt.grid(True)

# Loss1
plt.subplot(2, 2, 2)
plt.plot(steps, loss1, label="Loss1", color="orange", marker="o")
plt.xlabel("Step")
plt.ylabel("Loss1")
plt.title("Loss1")
plt.grid(True)

# Loss2
plt.subplot(2, 2, 3)
plt.plot(steps, loss2, label="Loss2", color="green", marker="o")
plt.xlabel("Step")
plt.ylabel("Loss2")
plt.title("Loss2")
plt.grid(True)

# Del Loss
plt.subplot(2, 2, 4)
plt.plot(steps, del_loss, label="Del Loss", color="red", marker="o")
plt.xlabel("Step")
plt.ylabel("Del Loss")
plt.title("Del Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_plots.png", dpi=300)
plt.show()

print("📊 图像已保存为 loss_plots.png")
