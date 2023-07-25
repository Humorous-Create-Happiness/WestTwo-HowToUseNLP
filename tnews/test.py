import json
import torch
from torch import LongTensor
from transformers import BertTokenizer, BertModel
from train import Classifier

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert = BertModel.from_pretrained("bert-base-chinese")
output_dim = 17  # 输出维度（分类数目）
print('finish loading bert!')

# 加载保存的模型
model = Classifier(bert, output_dim)
model.load_state_dict(torch.load("best_model2.pt"))
model.eval()
print('finish loading model!')

# 加载标签映射文件
with open("labels.json", "r", encoding="utf-8") as f:
    label_data = json.load(f)
label_mapping = {int(item["label"]): item["label_desc"] for item in label_data}

# 加载测试数据
with open("test_processed.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
print('finish loading json!')
# 预测并转换标签为label_desc
for item in test_data:
    sentence = item["sentence"]
    keywords = item["keywords"]

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
    step=0
    if step % 100 == 1:
        print(f"Step: {step}/{len(test_data)}")
    step += 1

    # 预测标签
    with torch.no_grad():
        output = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()
        predicted_label += 100
    # 转换为label_desc
    label_desc = label_mapping[predicted_label]

    # 将label_desc输出到keywords字段
    item["keywords"] = label_desc

# 保存结果到test.json文件
with open("test_processed_fin.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("预测结果已保存到test.json文件。")
