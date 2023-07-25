import json
import torch
from torch import LongTensor
from torch import nn
from torch import optim
from transformers import BertTokenizer, BertModel

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert = BertModel.from_pretrained("bert-base-chinese")
output_dim = 17  # 输出维度（分类数目）


# 加载数据集
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# 数据预处理
def preprocess_data(data):
    processed_data = []
    for item in data:
        label = int(item["label"])-100  #100-116,已经手动改了
        sentence = item["sentence"]

        # 使用tokenizer对句子进行编码
        encoded_data = tokenizer.encode_plus(
            sentence,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded_data["input_ids"].squeeze(0)
        attention_mask = encoded_data["attention_mask"].squeeze(0)

        processed_data.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        })
    return processed_data


# 创建分类模型
class Classifier(nn.Module):
    def __init__(self, bert, output_dim):
        super(Classifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        output = self.fc(output)
        return output


# 训练模型
def train_model(model, train_data, dev_data):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    best_acc = 0.0
    step=0
    for epoch in range(1):  # 迭代次数
        model.train()
        train_loss = 0.0
        train_correct = 0

        for item in train_data:
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)
            label = torch.tensor(item["label"]).unsqueeze(0).to(device)

            optimizer.zero_grad()

            output = model(input_ids, attention_mask)
            _, predicted = torch.max(output, 1)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)
            train_correct += (predicted == label).sum().item()

            if step%1000 ==1:
                print(f"Step: {step}/{len(train_data)}")
            step+=1
        train_loss /= len(train_data)
        train_acc = train_correct / len(train_data)

        # 在验证集上进行评估
        val_loss, val_acc = evaluate_model(model, dev_data)

        print(f"Epoch: {epoch + 1}/{10}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print()

        # 保存在验证集上表现最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model2.pt")

    print("Training finished.")
    print("Best validation accuracy: ", best_acc)


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
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)
            label = torch.tensor(item["label"]).unsqueeze(0).to(device)

            output = model(input_ids, attention_mask)
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
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)
            label = torch.tensor(item["label"]).unsqueeze(0).to(device)

            output = model(input_ids, attention_mask)
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
train_data = preprocess_data(train_data)

dev_data = load_data(dev_file)
dev_data = preprocess_data(dev_data)

#test_data = load_data(test_file)
#test_data = preprocess_data(test_data)

# 创建分类模型实例
model = Classifier(bert, output_dim)

# 训练模型
train_model(model, train_data, dev_data)

# 加载在验证集上表现最好的模型
model.load_state_dict(torch.load("best_model2.pt"))

# 评估测试集
#test_accuracy, test_results = evaluate_test_data(model, test_data)

# 打印测试集正确率
#print("Test Accuracy:", test_accuracy)

# 打印测试集结果
#print("Test Results:")
#print(test_results[0])
