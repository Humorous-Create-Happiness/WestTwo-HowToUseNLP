import json
import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer

# 加载字符集
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
char2idx = {c: i for i, c in enumerate(vocab)}
unknown_token = "[UNK]"  # 添加未知字符
char2idx[unknown_token] = len(vocab) - 1

# 加载数据集
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 数据预处理
def preprocess_data(data):
    processed_data = []
    for item in data:
        label = int(item["label"])
        sentence = item["sentence"]
        processed_data.append({
            "sentence": sentence,
            "label": label
        })
    return processed_data

# 创建分类模型
class Classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, output_dim):
        super(Classifier, self).__init__()
        self.transformer = Transformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, input_tensor):
        output_tensor = self.transformer(input_tensor)
        output = self.fc(output_tensor[:, 0, :])  # 取Transformer的输出的第一个位置的表示进行分类
        return output

# 加载训练数据和验证数据
train_file = "train_processed.json"  # 训练集文件路径
dev_file = "dev_processed.json"  # 验证集文件路径

train_data = load_data(train_file)
dev_data = load_data(dev_file)

# 模型参数
vocab_size = 6062  # 词汇表大小
embed_dim = 512  # 词向量维度
num_heads = 4  # 多头注意力头数
hidden_dim = 256  # 隐藏层维度
num_layers = 4  # 网络层数
output_dim = 119  # 输出维度（类别数目）

# 数据预处理
train_data = preprocess_data(train_data)
dev_data = preprocess_data(dev_data)

# 创建分类模型实例
model = Classifier(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, output_dim)

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
num_epochs = 10  # 迭代次数
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for item in train_data:
        sentence = item["sentence"]
        label = torch.tensor(item["label"]).unsqueeze(0).to(device)

        input_seq = [char2idx.get(c, char2idx[unknown_token]) for c in sentence]
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)

        optimizer.zero_grad()

        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += label.size(0)
        correct += (predicted == label).sum().item()

    # 打印训练信息
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss / len(train_data):.4f} - Accuracy: {correct / total:.4f}")

# 在验证集上评估模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for item in dev_data:
        sentence = item["sentence"]
        label = torch.tensor(item["label"]).unsqueeze(0).to(device)

        input_seq = [char2idx.get(c, char2idx[unknown_token]) for c in sentence]
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)

        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)

        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.4f}")
