import json
import torch
from torch import nn
from transformers import BertTokenizer
from model import Transformer

# 加载字符集
# 加载数据集
dataset = [
    "[BEGIN]事实证明 8M 参[MASK]就能做[MASK]差强人意的模型出来。[END]",
    "[BEGIN]这是另一个例子，你可以根据需要添加更多的句子。[END]"
]
with open('234.txt', 'r', encoding='utf-8') as f:  # 与训练时一致
    lines = f.readlines()
    for line in lines:
        dataset.append(line.strip())
print('finish loading dataset!')

# 构建词汇表---------------------------------------------------
vocab = set()
for sentence in dataset:
    vocab.update(list(sentence))
vocab = list(vocab)
vocab_size = 6062

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f)
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

print("词汇表加载完成！")  # 输出vocab_size的大小
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for i, c in enumerate(vocab)}
# 添加未知字符
unknown_token = "[UNK]"
vocab.append(unknown_token)
char2idx[unknown_token] = len(vocab) - 1
idx2char[len(vocab) - 1] = unknown_token
unknown_token_idx = char2idx[unknown_token]


# 加载数据集
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# 问题检查
def check_data_labels(data):
    missing_labels = []
    for i, item in enumerate(data):
        if "label" not in item:
            missing_labels.append(i)
    return missing_labels


# 数据预处理
def preprocess_data(data, tokenizer):
    processed_data = []
    for item in data:
        label = int(item["label"])           #______________________________________
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]

        # 使用tokenizer对句子进行编码
        encoded_data = tokenizer.encode_plus(
            sentence1,
            sentence2,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded_data["input_ids"].squeeze(0)
        attention_mask = encoded_data["attention_mask"].squeeze(0)

        # 确保input_ids和attention_mask的值不超出词汇表的索引范围
        input_ids = torch.clamp(input_ids, max=vocab_size - 1)
        attention_mask = torch.clamp(attention_mask, max=1)

        processed_data.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        })
    return processed_data


# 创建分类模型
class Classifier(nn.Module):
    def __init__(self, model, output_dim):
        super(Classifier, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(model.embed.embedding_dim, output_dim)

    def forward(self, input_ids):
        x = self.model.embed(input_ids)  # 将文本转换为整数索引序列
        x = self.model.positional_encoding(x)
        for transformer_block in self.model.transformer_blocks:
            x = transformer_block(x)
        x = x.mean(dim=1)  # 取平均作为句子表示
        output = self.dropout(x)
        output = self.fc(output)
        return output


# 训练模型
def train_model(model, train_data, dev_data):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    best_acc = 0.0

    for epoch in range(3):
        print(f"Epoch: {epoch + 1}/3")
        model.train()
        train_loss = 0.0
        train_correct = 0
        step = 0
        for item in train_data:
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            label = torch.tensor(item["label"]).unsqueeze(0).to(device)

            optimizer.zero_grad()

            output = model(input_ids)
            _, predicted = torch.max(output, 1)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)
            train_correct += (predicted == label).sum().item()
            if step % 1000 == 1:
                print(f"Step: {step}/{len(train_data)}")
            step += 1
        train_loss /= len(train_data)
        train_acc = train_correct / len(train_data)

        val_loss, val_acc = evaluate_model(model, dev_data)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_self.pth")

    print("Training finished.")
    print("Best validation accuracy:", best_acc)


# 在验证集上评估模型
def evaluate_model(model, data):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.eval()

    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for item in data:
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            label = torch.tensor(item["label"]).unsqueeze(0).to(device)

            output = model(input_ids)
            _, predicted = torch.max(output, 1)

            loss = nn.CrossEntropyLoss()(output, label)

            total_loss += loss.item() * input_ids.size(0)
            total_correct += (predicted == label).sum().item()

    avg_loss = total_loss / len(data)
    accuracy = total_correct / len(data)

    return avg_loss, accuracy


# 加载测试集并进行评估
def evaluate_test_data(model, test_data):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.eval()

    total_correct = 0

    results = []

    with torch.no_grad():
        for item in test_data:
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            label = torch.tensor(item["label"]).unsqueeze(0).to(device)

            output = model(input_ids)
            _, predicted = torch.max(output, 1)

            results.append({
                "sentence": item["sentence"],
                "predicted_label": predicted.item()
            })

            total_correct += (predicted == label).sum().item()

    accuracy = total_correct / len(test_data)

    return accuracy, results


# 设置文件路径
train_file = "train_processed.json"  # 训练集文件路径
dev_file = "dev_processed.json"  # 验证集文件路径
test_file = "test_processed.json"  # 测试集文件路径

# 加载并预处理数据
train_data = load_data(train_file)
dev_data = load_data(dev_file)
# test_data = load_data(test_file)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
train_data = preprocess_data(train_data, tokenizer)
dev_data = preprocess_data(dev_data, tokenizer)
# test_data = preprocess_data(test_data, tokenizer)

# 创建模型实例
vocab_size = 6062
embed_dim = 512
num_heads = 4
hidden_dim = 256
num_layers = 4
model = Transformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)
classifier = Classifier(model, output_dim=2)

# 训练模型
train_model(classifier, train_data, dev_data)

# 加载在验证集上表现最好的模型
classifier.load_state_dict(torch.load("best_model_self.pth"))

# 评估测试集
# test_accuracy, test_results = evaluate_test_data(classifier, test_data)

# 打印测试集正确率
# print("Test Accuracy:", test_accuracy)

# 打印测试集结果
print("Test Results:")
# print(test_results[0])
